// speaker-embedding-extractor-barra-impl.h — Sprecher-Embeddings auf der Tensor-G3-TPU.
//
// Ersetzt den ONNX-Embedding-Extractor durch den barra-Pfad: sherpa berechnet die
// kaldi-fbanks wie gehabt (Stream), der ResNet34-Trunk laeuft als vorkompiliertes
// TPU-Package (tpud, libbarra Zero-Copy), Statistik-Pooling + FC-Kopf in float auf
// der CPU (Massive-Activations-Muster wie beim Whisper-Glue).
//
// Aktivierung: env BARRA_EMB=1 (Weiche in SpeakerEmbeddingExtractorImpl::Create;
// das ONNX-Modell wird dann NICHT geladen). Konfiguration:
//   BARRA_EMB_HEAD  head.bin (Format s.u.; Default /root/pya/head.bin)
//   BARRA_SOCK_DIR  Socket-Verzeichnis des tpud mit dem Trunk-Package
//                   (Default /opt/hwbridge — fuer pyannote i.d.R. /opt/hwbridge/pya)
//
// head.bin: u32 magic 0x42454831 ("BEH1"), u32 dim (256), u32 stat (5120),
//           f64 in_scale, f64 out_scale, f32 W[dim*stat], f32 b[dim], f32 mv[dim].
//
// Segmente beliebiger Laenge: T<=300 Frames -> ein Fenster, zyklisch aufgefuellt
// (Frame-Wiederholung); T>300 -> nicht ueberlappende 300er-Fenster, letztes ans
// Segmentende geankert. Statistik (mean/var) wird ueber die Zeitspalten ALLER
// Fenster akkumuliert — das entspricht dem Full-Length-Pooling des Originals bis
// auf Conv-Randeffekte an den Fenstergrenzen.

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_BARRA_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_BARRA_IMPL_H_

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "Eigen/Dense"  // NOLINT
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"

extern "C" {
#include "sherpa-onnx/csrc/barra.h"
}

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorBarraImpl : public SpeakerEmbeddingExtractorImpl {
 public:
  static constexpr int32_t kFrames = 300;   // Fensterlaenge (3 s a 10 ms)
  static constexpr int32_t kFeat = 80;      // mel bins
  static constexpr int32_t kF = 10, kT = 38;  // Trunk-FM [C,F,T]; C kommt aus head.bin (stat/kF)

  explicit SpeakerEmbeddingExtractorBarraImpl(
      const SpeakerEmbeddingExtractorConfig & /*config*/) {
    const char *hp = std::getenv("BARRA_EMB_HEAD");
    if (!hp || !hp[0]) hp = "/root/pya/head.bin";
    // Modus-Erkennung VOR dem Kopf-Laden: titanet hat KEIN head.bin (Emb kommt komplett aus dem Tail)
    std::string dir(hp);
    auto sl = dir.find_last_of('/');
    dir = (sl == std::string::npos) ? "." : dir.substr(0, sl);
    tita_ = FileExists(dir + "/tita_tail.onnx");
    if (!tita_ && !LoadHead(hp)) {
      SHERPA_ONNX_LOGE("barra-emb: head.bin nicht ladbar: %s", hp);
      SHERPA_ONNX_EXIT(-1);
    }
    // eres2net-Modus: Kit traegt eres_tail.onnx (AFF-Fusionsmodule laufen exakt in float via Ort;
    // die TPU rechnet den Multi-Output-Rumpf — die Module selbst vertragen 16x8 nicht)
    tail_path_ = tita_ ? dir + "/tita_tail.onnx" : dir + "/eres_tail.onnx";
    eres_ = !tita_ && FileExists(tail_path_);
    if (barra_tpu_open(&tpu_) != 0) {
      SHERPA_ONNX_LOGE("barra-emb: tpud nicht erreichbar (BARRA_SOCK_DIR?)");
      SHERPA_ONNX_EXIT(-1);
    }
    if (tita_) {
      if (!LoadTitaGlue(dir + "/tita_glue.bin") || !LoadTitaParams(dir + "/tita_params.txt")) {
        SHERPA_ONNX_LOGE("barra-emb: tita_glue.bin/tita_params.txt fehlt/unlesbar");
        SHERPA_ONNX_EXIT(-1);
      }
      // 5 Segmente: mids 0-4; zbufs je Segment anlegen (Groessen via info2)
      for (int s = 0; s < 5; ++s) {
        uint32_t nin = 0, nout = 0, isz[4] = {0}, oszs[4] = {0};
        // nout MUSS in tita_osz_/tita_out_ ([5][2]) passen — sonst OOB-Schreiben in
        // benachbarte Member. isz[0] fuer Segment 0 gegen die bekannte Eingabegroesse
        // (kFrames*kFeat int16) pruefen; ein falscher Package-Satz ueberliefe sonst
        // beim quant_in das dmabuf-Mapping.
        if (barra_tpu_info2(&tpu_, (uint32_t)s, &nin, &nout, isz, oszs, 4) != 0 ||
            nin != 1 || nout < 1 || nout > 2 || isz[0] == 0 ||
            (s == 0 && isz[0] < (uint32_t)(kFrames * kFeat * 2))) {
          SHERPA_ONNX_LOGE("barra-emb: tita-Segment %d passt nicht (nin=%u nout=%u in=%u)",
                           s, nin, nout, isz[0]);
          SHERPA_ONNX_EXIT(-1);
        }
        tita_nout_[s] = (int)nout;
        if (barra_zc_alloc(&tita_in_[s], isz[0]) != 0) { SHERPA_ONNX_EXIT(-1); }
        for (uint32_t j = 0; j < nout; ++j) {
          tita_osz_[s][j] = oszs[j];
          if (barra_zc_alloc(&tita_out_[s][j], oszs[j]) != 0) { SHERPA_ONNX_EXIT(-1); }
        }
      }
      ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "tita-tail");
      Ort::SessionOptions so;
      so.SetIntraOpNumThreads(2);
      tail_ = std::make_unique<Ort::Session>(*ort_env_, tail_path_.c_str(), so);
      {
        Ort::AllocatorWithDefaultOptions alloc;
        tail_in_names_.push_back(tail_->GetInputNameAllocated(0, alloc).get());
        auto shp = tail_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        tail_in_shapes_.push_back(shp);
        tail_out_name_ = tail_->GetOutputNameAllocated(0, alloc).get();
        auto os = tail_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        dim_ = (int32_t)os.back();
      }
      SHERPA_ONNX_LOGE("barra-emb: TPU-Embedding aktiv (titanet, 5 Segmente, dim=%d)", (int)dim_);
      return;
    }
    if (eres_) {
      uint32_t nin = 0, nout = 0, isz[4] = {0}, oszs[4] = {0};
      if (barra_tpu_info2(&tpu_, 0, &nin, &nout, isz, oszs, 4) != 0 ||
          nin != 1 || nout != 3 || isz[0] != kFeat * kFrames * 2) {
        SHERPA_ONNX_LOGE("barra-emb: eres-Rumpf passt nicht (nin=%u nout=%u in=%u)",
                         nin, nout, isz[0]);
        SHERPA_ONNX_EXIT(-1);
      }
      if (barra_zc_alloc(&in_, isz[0]) != 0) { SHERPA_ONNX_EXIT(-1); }
      for (int i = 0; i < 3; ++i) {
        osz_[i] = oszs[i];
        if (barra_zc_alloc(&outs_[i], oszs[i]) != 0) { SHERPA_ONNX_EXIT(-1); }
      }
      if (!LoadEresParams(dir + "/eres_params.txt")) {
        SHERPA_ONNX_LOGE("barra-emb: eres_params.txt fehlt/unlesbar");
        SHERPA_ONNX_EXIT(-1);
      }
      ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "eres-tail");
      Ort::SessionOptions so;
      so.SetIntraOpNumThreads(2);
      tail_ = std::make_unique<Ort::Session>(*ort_env_, tail_path_.c_str(), so);
      for (size_t i = 0; i < tail_->GetInputCount(); ++i) {
        Ort::AllocatorWithDefaultOptions alloc;
        tail_in_names_.push_back(tail_->GetInputNameAllocated(i, alloc).get());
        auto shp = tail_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        tail_in_shapes_.push_back(shp);
        int64_t n = 1; for (auto d : shp) n *= d;
        tail_in_sizes_.push_back(n);
      }
      {
        Ort::AllocatorWithDefaultOptions alloc;
        tail_out_name_ = tail_->GetOutputNameAllocated(0, alloc).get();
      }
      SHERPA_ONNX_LOGE("barra-emb: TPU-Embedding aktiv (eres2net, dim=%d, C=%d)",
                       (int)dim_, (int)C_);
    } else {
      uint32_t isz = 0, osz = 0, nm = 0;
      if (barra_tpu_info(&tpu_, 0, &isz, &osz, &nm) != 0 ||
          isz != kFeat * kFrames * 2 || osz != (uint32_t)(C_ * kF * kT * 2)) {
        SHERPA_ONNX_LOGE("barra-emb: Modell-0 I/O passt nicht (in=%u out=%u)",
                         isz, osz);
        SHERPA_ONNX_EXIT(-1);
      }
      if (barra_zc_alloc(&in_, isz) != 0 || barra_zc_alloc(&out_, osz) != 0) {
        SHERPA_ONNX_LOGE("barra-emb: dmabuf-Alloc fehlgeschlagen");
        SHERPA_ONNX_EXIT(-1);
      }
      SHERPA_ONNX_LOGE("barra-emb: TPU-Embedding aktiv (dim=%d, head=%s)",
                       (int)dim_, hp);
    }
  }

  ~SpeakerEmbeddingExtractorBarraImpl() override {
    barra_zc_free(&in_);
    barra_zc_free(&out_);
    for (int i = 0; i < 3; ++i) barra_zc_free(&outs_[i]);
    for (int s = 0; s < 5; ++s) {
      barra_zc_free(&tita_in_[s]);
      for (int j = 0; j < 2; ++j) barra_zc_free(&tita_out_[s][j]);
    }
    barra_tpu_close(&tpu_);
  }

  int32_t Dim() const override { return dim_; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    feat_config.sampling_rate = 16000;
    if (tita_) {
      // NeMo-Konvention (vgl. speaker-embedding-extractor-nemo-impl.h)
      feat_config.feature_dim = kFeat;
      feat_config.normalize_samples = true;
      feat_config.snip_edges = true;
      feat_config.low_freq = 0;
      feat_config.is_librosa = true;
      feat_config.remove_dc_offset = false;
      feat_config.window_type = "hann";
    } else {
      feat_config.normalize_samples = false;  // wespeaker: kaldi-PCM-Skala
    }
    return std::make_unique<OnlineStream>(feat_config);
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() < s->NumFramesReady();
  }

  std::vector<float> Compute(OnlineStream *s) const override {
    int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
    if (num_frames <= 0) {
      SHERPA_ONNX_LOGE("barra-emb: keine Frames bereit");
      return {};
    }
    std::vector<float> features =
        s->GetFrames(s->GetNumProcessedFrames(), num_frames);
    s->GetNumProcessedFrames() += num_frames;
    const int32_t T = num_frames;
    if ((int32_t)features.size() != T * kFeat) {
      SHERPA_ONNX_LOGE("barra-emb: feat_dim != %d", kFeat);
      return {};
    }

    // Fensterstarts: ein Fenster (mit Wrap) oder 300er-Kacheln + Endanker
    std::vector<int32_t> starts;
    if (T <= kFrames) {
      starts.push_back(0);
    } else {
      for (int32_t s0 = 0; s0 + kFrames <= T; s0 += kFrames) starts.push_back(s0);
      if (starts.back() + kFrames < T) starts.push_back(T - kFrames);
    }

    if (tita_) {
      // titanet: je Fenster per_feature-Norm -> 5-Segment-TPU-Kette + SE-Glue -> Ort-Tail -> Emb;
      // mehrere Fenster werden gemittelt
      std::lock_guard<std::mutex> lk(mu_);
      std::vector<double> acc(dim_, 0.0);
      std::vector<float> win((size_t)kFrames * kFeat);
      for (int32_t s0 : starts) {
        for (int32_t t = 0; t < kFrames; ++t) {
          int32_t st = (T <= kFrames) ? (t % T) : (s0 + t);
          for (int32_t f = 0; f < kFeat; ++f)
            win[(size_t)t * kFeat + f] = features[(size_t)st * kFeat + f];
        }
        for (int32_t f = 0; f < kFeat; ++f) {   // per_feature-Norm ueber das Fenster
          double mu2 = 0, s2 = 0;
          for (int32_t t = 0; t < kFrames; ++t) mu2 += win[(size_t)t * kFeat + f];
          mu2 /= kFrames;
          for (int32_t t = 0; t < kFrames; ++t) { double d = win[(size_t)t * kFeat + f] - mu2; s2 += d * d; }
          double sd = std::sqrt(s2 / kFrames) + 1e-5;
          for (int32_t t = 0; t < kFrames; ++t)
            win[(size_t)t * kFeat + f] = (float)((win[(size_t)t * kFeat + f] - mu2) / sd);
        }
        std::vector<float> emb1;
        if (!RunTitaChain(win.data(), &emb1)) return {};
        for (int32_t i = 0; i < dim_; ++i) acc[i] += emb1[i];
      }
      std::vector<float> ans(dim_);
      for (int32_t i = 0; i < dim_; ++i) ans[i] = (float)(acc[i] / starts.size());
      return ans;
    }

    std::vector<double> sum((size_t)C_ * kF, 0.0), sq((size_t)C_ * kF, 0.0);
    std::vector<float> fm((size_t)C_ * kF * kT);
    int64_t ncols = 0;

    std::lock_guard<std::mutex> lk(mu_);
    for (int32_t s0 : starts) {
      auto *ip = reinterpret_cast<int16_t *>(in_.map);
      for (int32_t f = 0; f < kFeat; ++f) {
        for (int32_t t = 0; t < kFrames; ++t) {
          int32_t st = (T <= kFrames) ? (t % T) : (s0 + t);
          float v = features[(size_t)st * kFeat + f] / (float)in_scale_;
          int32_t q = (int32_t)lrintf(v);
          if (q > 32767) q = 32767;
          if (q < -32768) q = -32768;
          ip[(size_t)f * kFrames + t] = (int16_t)q;
        }
      }
      if (eres_) {
        if (!RunEres(fm.data())) return {};
      } else {
        uint32_t us = 0;
        if (barra_tpu_infer(const_cast<barra_tpu *>(&tpu_), 0,
                            const_cast<barra_zbuf *>(&in_),
                            const_cast<barra_zbuf *>(&out_), &us) != 0) {
          SHERPA_ONNX_LOGE("barra-emb: TPU-Inferenz fehlgeschlagen");
          return {};
        }
        const auto *op = reinterpret_cast<const int16_t *>(out_.map);
        for (size_t i = 0; i < fm.size(); ++i) fm[i] = (float)(op[i] * out_scale_);
      }
      for (int32_t cf = 0; cf < C_ * kF; ++cf) {
        const float *row = fm.data() + (size_t)cf * kT;
        double a = 0.0, a2 = 0.0;
        for (int32_t t = 0; t < kT; ++t) { a += row[t]; a2 += (double)row[t] * row[t]; }
        sum[cf] += a;
        sq[cf] += a2;
      }
      ncols += kT;
    }

    // Statistik-Pooling (Bessel) + FC-Kopf, alles double-Akkumulation
    const int32_t stat = C_ * kF;
    std::vector<double> x((size_t)stat * 2);
    for (int32_t i = 0; i < stat; ++i) {
      double m = sum[i] / (double)ncols;
      double var = (sq[i] / (double)ncols - m * m) * (double)ncols /
                   (double)(ncols - 1);
      if (var < 0) var = 0;
      x[i] = m;
      x[(size_t)stat + i] = std::sqrt(var + 1e-8);
    }
    std::vector<float> ans(dim_);
    for (int32_t r = 0; r < dim_; ++r) {
      const float *wr = W_.data() + (size_t)r * stat * 2;
      double acc = 0.0;
      for (int32_t i = 0; i < stat * 2; ++i) acc += (double)wr[i] * x[i];
      ans[r] = (float)(acc + b_[r] - mv_[r]);
    }
    return ans;
  }

 private:
  bool LoadHead(const char *path) {
    FILE *fp = std::fopen(path, "rb");
    if (!fp) return false;
    uint32_t magic = 0, dim = 0, stat = 0;
    bool ok = std::fread(&magic, 4, 1, fp) == 1 &&
              std::fread(&dim, 4, 1, fp) == 1 &&
              std::fread(&stat, 4, 1, fp) == 1 && magic == 0x42454831u &&
              dim > 0 && stat > 0 && stat % kF == 0;
    if (ok) {
      dim_ = (int32_t)dim;
      C_ = (int32_t)(stat / kF);   // r34: 256, eres2net: 512
      W_.resize((size_t)dim * stat * 2);
      b_.resize(dim);
      mv_.resize(dim);
      ok = std::fread(&in_scale_, 8, 1, fp) == 1 &&
           std::fread(&out_scale_, 8, 1, fp) == 1 &&
           std::fread(W_.data(), 4, W_.size(), fp) == W_.size() &&
           std::fread(b_.data(), 4, b_.size(), fp) == b_.size() &&
           std::fread(mv_.data(), 4, mv_.size(), fp) == mv_.size();
    }
    std::fclose(fp);
    return ok;
  }

  bool LoadTitaGlue(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    uint32_t magic = 0, nb = 0;
    bool ok = std::fread(&magic, 4, 1, fp) == 1 && std::fread(&nb, 4, 1, fp) == 1 &&
              magic == 0x42544731u && nb == 5;
    for (uint32_t b = 0; ok && b < nb; ++b) {
      uint32_t sk = 0, cin = 0, chid = 0;
      ok = std::fread(&sk, 4, 1, fp) == 1 && std::fread(&cin, 4, 1, fp) == 1 &&
           std::fread(&chid, 4, 1, fp) == 1;
      if (!ok) break;
      tg_skip_[b] = (int)sk; tg_cin_[b] = (int)cin; tg_chid_[b] = (int)chid;
      tg_fc1_[b].resize((size_t)cin * chid);
      tg_fc2_[b].resize((size_t)chid * cin);
      ok = std::fread(tg_fc1_[b].data(), 4, tg_fc1_[b].size(), fp) == tg_fc1_[b].size() &&
           std::fread(tg_fc2_[b].data(), 4, tg_fc2_[b].size(), fp) == tg_fc2_[b].size();
    }
    std::fclose(fp);
    return ok;
  }

  bool LoadTitaParams(const std::string &path) {
    FILE *fp = std::fopen(path.c_str(), "r");
    if (!fp) return false;
    char line[128];
    while (std::fgets(line, sizeof line, fp)) {
      int s = 0, o = 0; double v = 0; long zp = 0;
      if (std::sscanf(line, "s%d_isc=%lf", &s, &v) == 2 && s < 5) tp_isc_[s] = v;
      else if (std::sscanf(line, "s%d_o%d_osc=%lf", &s, &o, &v) == 3 && s < 5 && o < 2) tp_osc_[s][o] = v;
      else if (std::sscanf(line, "s%d_o%d_ozp=%ld", &s, &o, &zp) == 3 && s < 5 && o < 2) tp_ozp_[s][o] = (int)zp;
    }
    std::fclose(fp);
    for (int s = 0; s < 5; ++s) if (!(tp_isc_[s] > 0)) return false;
    return true;
  }

  // SE-Glue: y = relu(x*sigmoid(relu(mean(x)@fc1)@fc2) [+ skip]); x/skip/y: [T,C] T-major (Eigen)
  void TitaGlue(int b, const float *x, const float *skip, float *y, int C) const {
    using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    int H = tg_chid_[b];
    Eigen::Map<const RM> X(x, kFrames, C);
    Eigen::Map<const RM> F1(tg_fc1_[b].data(), C, H);
    Eigen::Map<const RM> F2(tg_fc2_[b].data(), H, C);
    Eigen::RowVectorXf mean = X.colwise().mean();
    Eigen::RowVectorXf h = (mean * F1).cwiseMax(0.0f);
    Eigen::RowVectorXf gate = (1.0f + (-(h * F2).array()).exp()).inverse();
    Eigen::Map<RM> Y(y, kFrames, C);
    if (skip) {
      Eigen::Map<const RM> S(skip, kFrames, C);
      Y = ((X.array().rowwise() * gate.array()) + S.array()).cwiseMax(0.0f);
    } else {
      Y = (X.array().rowwise() * gate.array()).cwiseMax(0.0f);
    }
  }

  bool RunTitaChain(const float *win, std::vector<float> *emb) const {
    auto quant_in = [&](int s, const float *src, int n) {
      auto *ip = reinterpret_cast<int16_t *>(tita_in_[s].map);
      double isc = tp_isc_[s];
      for (int i = 0; i < n; ++i) {
        int32_t q = (int32_t)lrint(src[i] / isc);
        if (q > 32767) q = 32767; if (q < -32768) q = -32768;
        ip[i] = (int16_t)q;
      }
    };
    auto deq_out = [&](int s, int o, std::vector<float> *dst) {
      int n = (int)(tita_osz_[s][o] / 2);
      dst->resize(n);
      const auto *op = reinterpret_cast<const int16_t *>(tita_out_[s][o].map);
      double osc = tp_osc_[s][o]; int ozp = tp_ozp_[s][o];
      for (int i = 0; i < n; ++i) (*dst)[i] = (float)((op[i] - ozp) * osc);
    };
    auto infer = [&](int s) {
      uint32_t us = 0;
      barra_zbuf *ins[1] = {const_cast<barra_zbuf *>(&tita_in_[s])};
      barra_zbuf *outs[2] = {const_cast<barra_zbuf *>(&tita_out_[s][0]),
                             const_cast<barra_zbuf *>(&tita_out_[s][1])};
      return barra_tpu_infer_multi(const_cast<barra_tpu *>(&tpu_), (uint32_t)s,
                                   ins, 1, outs, tita_nout_[s], &us) == 0;
    };
    std::vector<float> x, skip, y;
    quant_in(0, win, kFrames * kFeat);
    if (!infer(0)) return false;
    deq_out(0, 0, &x);
    y.resize(x.size());
    TitaGlue(0, x.data(), nullptr, y.data(), (int)(x.size() / kFrames));
    for (int s = 1; s <= 3; ++s) {
      quant_in(s, y.data(), (int)y.size());
      if (!infer(s)) return false;
      deq_out(s, 0, &x); deq_out(s, 1, &skip);
      y.resize(x.size());
      TitaGlue(s, x.data(), skip.data(), y.data(), (int)(x.size() / kFrames));
    }
    quant_in(4, y.data(), (int)y.size());
    if (!infer(4)) return false;
    deq_out(4, 0, &x);                       // [T, 3072] T-major
    int C = (int)(x.size() / kFrames);
    y.resize(x.size());
    TitaGlue(4, x.data(), nullptr, y.data(), C);
    // Tail erwartet NCHW [1, C, T]
    std::vector<float> nchw((size_t)C * kFrames);
    for (int t = 0; t < kFrames; ++t)
      for (int c = 0; c < C; ++c) nchw[(size_t)c * kFrames + t] = y[(size_t)t * C + c];
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto tin = Ort::Value::CreateTensor(mi, nchw.data(), nchw.size(),
                                        tail_in_shapes_[0].data(), tail_in_shapes_[0].size());
    const char *in_names[1] = {tail_in_names_[0].c_str()};
    const char *out_names[1] = {tail_out_name_.c_str()};
    auto out = const_cast<Ort::Session *>(tail_.get())->Run(Ort::RunOptions{nullptr},
        in_names, &tin, 1, out_names, 1);
    const float *p = out[0].GetTensorData<float>();
    emb->assign(p, p + dim_);
    return true;
  }

  static bool FileExists(const std::string &p) {
    FILE *f = std::fopen(p.c_str(), "rb");
    if (f) { std::fclose(f); return true; }
    return false;
  }

  bool LoadEresParams(const std::string &path) {
    // key=value: in_isc + osc_<bytes> je Rumpf-Output (Zuordnung ueber die Output-Groesse)
    FILE *fp = std::fopen(path.c_str(), "r");
    if (!fp) return false;
    char line[128];
    while (std::fgets(line, sizeof line, fp)) {
      double v = 0; unsigned long sz = 0;
      if (std::sscanf(line, "in_isc=%lf", &v) == 1) eres_isc_ = v;
      else if (std::sscanf(line, "osc_%lu=%lf", &sz, &v) == 2) {
        for (int i = 0; i < 3; ++i) if (osz_[i] == (uint32_t)sz) eres_osc_[i] = v;
      }
    }
    std::fclose(fp);
    bool ok = eres_isc_ > 0;
    for (int i = 0; i < 3; ++i) ok = ok && eres_osc_[i] > 0;
    // Der eres-Rumpf ist mit derselben fbank-Skala kalibriert wie der Kopf sie erwartet
    if (ok) in_scale_ = eres_isc_;
    return ok;
  }

  bool RunEres(float *fm) const {
    uint32_t us = 0;
    barra_zbuf *ins[1] = {const_cast<barra_zbuf *>(&in_)};
    barra_zbuf *outs[3] = {const_cast<barra_zbuf *>(&outs_[0]),
                           const_cast<barra_zbuf *>(&outs_[1]),
                           const_cast<barra_zbuf *>(&outs_[2])};
    if (barra_tpu_infer_multi(const_cast<barra_tpu *>(&tpu_), 0, ins, 1, outs, 3, &us) != 0) {
      SHERPA_ONNX_LOGE("barra-emb: eres-Multi-Inferenz fehlgeschlagen");
      return false;
    }
    // Outputs dequantisieren; Zuordnung zu den Tail-Inputs ueber die Elementzahl
    // (fuse12 1536000B/768000 el, layer3 768000B/384000 el, layer4 389120B/194560 el — eindeutig)
    std::vector<Ort::Value> tin;
    Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<std::vector<float>> bufs(tail_in_names_.size());
    std::vector<const char *> in_names;
    for (size_t i = 0; i < tail_in_names_.size(); ++i) {
      int64_t want = tail_in_sizes_[i];
      int src = -1;
      for (int k = 0; k < 3; ++k) if ((int64_t)(osz_[k] / 2) == want) src = k;
      if (src < 0) { SHERPA_ONNX_LOGE("barra-emb: Tail-Input %zu ohne Rumpf-Output", i); return false; }
      bufs[i].resize((size_t)want);
      const auto *op = reinterpret_cast<const int16_t *>(outs_[src].map);
      const double sc = eres_osc_[src];
      for (int64_t j = 0; j < want; ++j) bufs[i][(size_t)j] = (float)(op[j] * sc);
      tin.push_back(Ort::Value::CreateTensor(mi, bufs[i].data(), bufs[i].size(),
                                             tail_in_shapes_[i].data(), tail_in_shapes_[i].size()));
      in_names.push_back(tail_in_names_[i].c_str());
    }
    const char *out_names[1] = {tail_out_name_.c_str()};
    auto out = const_cast<Ort::Session *>(tail_.get())->Run(Ort::RunOptions{nullptr},
        in_names.data(), tin.data(), tin.size(), out_names, 1);
    const float *p = out[0].GetTensorData<float>();
    std::copy(p, p + (size_t)C_ * kF * kT, fm);
    return true;
  }

  int32_t dim_ = 0;
  int32_t C_ = 256;
  bool eres_ = false;
  bool tita_ = false;
  int tita_nout_[5] = {0};
  uint32_t tita_osz_[5][2] = {{0}};
  double tp_isc_[5] = {0}, tp_osc_[5][2] = {{0}};
  int tp_ozp_[5][2] = {{0}};
  int tg_skip_[5] = {0}, tg_cin_[5] = {0}, tg_chid_[5] = {0};
  std::vector<float> tg_fc1_[5], tg_fc2_[5];
  mutable barra_zbuf tita_in_[5] = {{-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1},
                                    {-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1},
                                    {-1, nullptr, 0, -1, -1, -1}};
  mutable barra_zbuf tita_out_[5][2] = {{{-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1}},
                                        {{-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1}},
                                        {{-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1}},
                                        {{-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1}},
                                        {{-1, nullptr, 0, -1, -1, -1}, {-1, nullptr, 0, -1, -1, -1}}};
  double eres_isc_ = 0.0;
  double eres_osc_[3] = {0.0, 0.0, 0.0};
  uint32_t osz_[3] = {0, 0, 0};
  std::string tail_path_, tail_out_name_;
  std::vector<std::string> tail_in_names_;
  std::vector<std::vector<int64_t>> tail_in_shapes_;
  std::vector<int64_t> tail_in_sizes_;
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> tail_;
  mutable barra_zbuf outs_[3] = {{-1, nullptr, 0, -1, -1, -1},
                                 {-1, nullptr, 0, -1, -1, -1},
                                 {-1, nullptr, 0, -1, -1, -1}};
  double in_scale_ = 0.0, out_scale_ = 0.0;
  std::vector<float> W_, b_, mv_;
  mutable barra_tpu tpu_{-1, 0};
  mutable barra_zbuf in_{-1, nullptr, 0, -1, -1, -1};
  mutable barra_zbuf out_{-1, nullptr, 0, -1, -1, -1};
  mutable std::mutex mu_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_BARRA_IMPL_H_
