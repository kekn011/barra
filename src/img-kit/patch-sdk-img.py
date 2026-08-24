# -*- coding: utf-8 -*-
# SDK-Patch: Bild-Kit (img) in Setup-GUI, Core, i18n, .gitignore einhaengen. BOM/CRLF der .ps1 bleiben erhalten.
import io, re, os, sys
R = r'C:\Users\kevin\projects\barra'
def rd(p):
    b = open(p, 'rb').read()
    bom = b.startswith(b'\xef\xbb\xbf')
    s = b.decode('utf-8-sig')
    crlf = '\r\n' in s
    return s.replace('\r\n', '\n'), bom, crlf
def wr(p, s, bom, crlf):
    if crlf: s = s.replace('\n', '\r\n')
    open(p, 'wb').write((b'\xef\xbb\xbf' if bom else b'') + s.encode('utf-8'))
def edit(p, reps):
    s, bom, crlf = rd(p)
    for old, new, cnt in reps:
        c = s.count(old)
        if c != cnt:
            print('ANKER-FEHLER in %s: %r erwartet %d, gefunden %d' % (os.path.basename(p), old[:70], cnt, c)); sys.exit(1)
        s = s.replace(old, new)
    wr(p, s, bom, crlf); print('ok', os.path.basename(p), 'bom=%s crlf=%s' % (bom, crlf))

# ---------------- barra-setup.ps1 ----------------
P = os.path.join(R, r'barra-setup\barra-setup.ps1')
s, _, _ = rd(P)
if 'CardImg' in s:
    print('setup schon gepatcht')
else:
    card_wake_end = '''                <TextBlock x:Name="FactWake" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
'''
    card_img = card_wake_end + '''            <Border x:Name="CardImg" Background="#0E1116" BorderBrush="#262A33" BorderThickness="1" CornerRadius="12" Padding="18,14" Margin="0,0,0,12">
              <StackPanel>
                <CheckBox x:Name="ChkImg" Foreground="#F0F2F6" FontSize="15" FontWeight="SemiBold"/>
                <TextBlock x:Name="DescImg" Margin="26,6,0,0" TextWrapping="Wrap" FontSize="13"/>
                <TextBlock x:Name="FactImg" Style="{StaticResource Hint}" Margin="26,6,0,0"/>
              </StackPanel>
            </Border>
'''
    edit(P, [
        (card_wake_end, card_img, 1),
        ('            <CheckBox x:Name="PkgWake" Margin="0,6,0,0" Foreground="#C9CFD8"/>\n',
         '            <CheckBox x:Name="PkgWake" Margin="0,6,0,0" Foreground="#C9CFD8"/>\n            <CheckBox x:Name="PkgImg" Margin="0,6,0,0" Foreground="#C9CFD8"/>\n', 1),
        ("'PkgWake'", "'PkgWake','PkgImg'", 1),
        ("'FactWake'", "'FactWake','CardImg','ChkImg','DescImg','FactImg'", 1),
        ("$ui.PkgWake.Content=T 'setup.p3.pkg_wake';", "$ui.PkgWake.Content=T 'setup.p3.pkg_wake'; $ui.PkgImg.Content=T 'setup.p3.pkg_img';", 1),
        ("  $ui.ChkWake.Content=T 'setup.pkg.wake_name'; $ui.DescWake.Text=T 'setup.pkg.wake_desc'; $ui.FactWake.Text=T 'setup.pkg.wake_fact'\n",
         "  $ui.ChkWake.Content=T 'setup.pkg.wake_name'; $ui.DescWake.Text=T 'setup.pkg.wake_desc'; $ui.FactWake.Text=T 'setup.pkg.wake_fact'\n  $ui.ChkImg.Content=T 'setup.pkg.img_name'; $ui.DescImg.Text=T 'setup.pkg.img_desc'; $ui.FactImg.Text=T 'setup.pkg.img_fact'\n", 1),
        ("      @{ id='heybarra'; name={T 'setup.model.wake_heybarra'}; desc={T 'setup.model.wake_heybarra_desc'}; files=@('wake-kit\\wake-kit.tar.gz') } ) }\n",
         "      @{ id='heybarra'; name={T 'setup.model.wake_heybarra'}; desc={T 'setup.model.wake_heybarra_desc'}; files=@('wake-kit\\wake-kit.tar.gz') } ) }\n"
         "  @{ id='img'; name={T 'setup.pkg.img_name'}; models=@(\n"
         "      @{ id='dreamshaper-lcm'; name={T 'setup.model.img_lcm'}; desc={T 'setup.model.img_lcm_desc'}; files=@('img-kit\\img-kit.tar.gz','img-kit\\DreamShaper8_LCM.safetensors','img-kit\\taesd.safetensors') } ) }\n", 1),
        (",@('wake',$ui.CardWake,$ui.ChkWake))", ",@('wake',$ui.CardWake,$ui.ChkWake),@('img',$ui.CardImg,$ui.ChkImg))", 1),
        ("foreach ($cb in @($ui.ChkLlm,$ui.ChkStt,$ui.ChkPya,$ui.ChkTts,$ui.ChkWake))", "foreach ($cb in @($ui.ChkLlm,$ui.ChkStt,$ui.ChkPya,$ui.ChkTts,$ui.ChkWake,$ui.ChkImg))", 1),
        ("  if ($ui.ChkWake.IsEnabled -and $ui.ChkWake.IsChecked) { $sel+='wake' }\n",
         "  if ($ui.ChkWake.IsEnabled -and $ui.ChkWake.IsChecked) { $sel+='wake' }\n  if ($ui.ChkImg.IsEnabled -and $ui.ChkImg.IsChecked) { $sel+='img' }\n", 1),
        ("$script:kitWake=$false; $script:kitAvail=@()", "$script:kitWake=$false; $script:kitImg=$false; $script:kitAvail=@()", 1),
        ("  $script:kitWake = [bool](Get-KitModels 'wake').Count\n", "  $script:kitWake = [bool](Get-KitModels 'wake').Count\n  $script:kitImg = [bool](Get-KitModels 'img').Count\n", 1),
        ("  $ui.PkgWake.IsEnabled=$script:kitWake; $ui.PkgWake.IsChecked=$script:kitWake\n",
         "  $ui.PkgWake.IsEnabled=$script:kitWake; $ui.PkgWake.IsChecked=$script:kitWake\n  $ui.PkgImg.IsEnabled=$script:kitImg; $ui.PkgImg.IsChecked=$script:kitImg\n", 1),
        ("if ($ui.PkgWake.IsChecked) { $sel+='wake' }", "if ($ui.PkgWake.IsChecked) { $sel+='wake' }; if ($ui.PkgImg.IsChecked) { $sel+='img' }", 1),
        ("$ui.PkgWake.IsEnabled=$false\n", "$ui.PkgWake.IsEnabled=$false; $ui.PkgImg.IsEnabled=$false\n", 1),
        ("$ui.PkgWake.IsEnabled=$script:kitWake }", "$ui.PkgWake.IsEnabled=$script:kitWake; $ui.PkgImg.IsEnabled=$script:kitImg }", 1),
    ])

# ---------------- barra-core.ps1 ----------------
P = os.path.join(R, r'barra-setup\barra-core.ps1')
s, _, _ = rd(P)
if "$k -eq 'img'" in s:
    print('core schon gepatcht')
else:
    anchor = "    }\n    if ($bothKits) { Pkg (T 'core.kit.both_note') 'ok' } else { Pkg (T 'core.kit.all_ok') 'ok' }\n"
    block = '''      elseif ($k -eq 'img') {
        # Bildgenerator (Android-seitig wie llm/stt): Kit -> /data/local/barra-img/{bin,models}, imgserver.sh -> baseos/bin,
        # Container-CLI barra-img -> /usr/local/bin. Manueller Start (imgserver.sh start, Port 8096), KEIN Boot-Autostart.
        $name = T 'core.kit.img_name'
        $mdl = if ($km['img']) { $km['img'] } else { 'dreamshaper-lcm' }
        $mf = @{ 'dreamshaper-lcm'='DreamShaper8_LCM.safetensors' }[$mdl]
        if (-not $mf) { throw (T 'core.kit.fail' "$name (model id $mdl)") }
        Pkg (T 'core.kit.push' $name)
        KitPush (Join-Path $script:Kit 'img-kit\\img-kit.tar.gz') '/data/local/tmp/img-kit.tar.gz' $name
        Pkg (T 'core.kit.extract' $name)
        $r = AdbSuBg ($KitDiag + 'D=/data/local/barra-img; i=0; while [ $i -lt 60 ]; do mkdir -p $D && touch $D/.wt && rm -f $D/.wt && break; [ $i = 0 ] && kdiag erst; [ $((i%12)) = 11 ] && kdiag lauf; sleep 5; i=$((i+1)); done; mkdir -p $D && touch $D/.wt && rm -f $D/.wt || kdiag final; cd $D && rm -rf bin base && tar -xzf /data/local/tmp/img-kit.tar.gz && mkdir -p models && chmod -R 755 $D/bin && cp base/imgserver.sh /data/adb/baseos/bin/imgserver.sh && chmod 755 /data/adb/baseos/bin/imgserver.sh && U=/data/local/ubuntu && cp base/barra-img $U/usr/local/bin/barra-img && chmod 755 $U/usr/local/bin/barra-img && rm -rf base /data/local/tmp/img-kit.tar.gz') 900 (T 'core.kit.extract' $name)
        if (-not $r.ok) { throw (T 'core.kit.fail' "$name (tar rc=$($r.rc)): $($r.log)") }
        Pkg (T 'core.kit.push_model' $name)
        KitPush (Join-Path $script:Kit "img-kit\\$mf") '/data/local/tmp/img-model.bin' $name
        KitPush (Join-Path $script:Kit 'img-kit\\taesd.safetensors') '/data/local/tmp/img-taesd.bin' $name
        $o = AdbSuM ('D=/data/local/barra-img/models; mkdir -p $D && mv /data/local/tmp/img-model.bin $D/' + $mf + ' && mv /data/local/tmp/img-taesd.bin $D/taesd.safetensors && chmod 644 $D/* && echo MDL_OK') 180
        if ($o -notmatch 'MDL_OK') { throw (T 'core.kit.fail' "$name (model)") }
        # KEIN Auto-Start (Kevin): Start manuell via 'imgserver.sh start'
        Ok (T 'core.kit.ok' $name)
      }
'''
    edit(P, [(anchor, block + anchor, 1)])

# ---------------- i18n (Kit-Kopie + src) ----------------
DE = {
 'setup.pkg.img_name': 'Bildgenerator (Text-zu-Bild)',
 'setup.pkg.img_desc': 'Erzeugt Bilder aus Textbeschreibungen — vollständig auf dem Gerät, ohne Cloud. Stable Diffusion 1.5 (DreamShaper 8 mit LCM-Sampler) rechnet auf dem Grafikkern mit eigens für den Mali-Chip geschriebenen Rechenkernen; ein 512×512-Bild in 4 Schritten dauert rund 17 Sekunden. Abruf per HTTP oder im Container mit barra-img "…" — als Bildantwort im Zusammenspiel mit KI-Chat und Spracherkennung.',
 'setup.pkg.img_fact': '2,2 GB · HTTP-Dienst (Port 8096) · manueller Start (imgserver.sh), kein Boot-Autostart · belegt 1,9 GB Arbeitsspeicher, solange er läuft',
 'setup.model.img_lcm': 'DreamShaper 8 LCM (Stable Diffusion 1.5)',
 'setup.model.img_lcm_desc': 'Community-Modell DreamShaper 8 (Lykon) auf Stable-Diffusion-1.5-Basis mit einkalkuliertem LCM-Sampler: 4 Schritte statt 20–30, 2,1 GB (f16) plus TAESD-Schnelldekoder (10 MB). Gemessen auf dem Node: 512×512 in 16,5–17 s je Bild bei laufendem Dienst (3,8 s je Schritt; das erste Bild nach dem Start braucht ~20 s). Fotorealistisch bis illustrativ, versteht englische Beschreibungen am besten; mit Stock-Vulkan dauerte dasselbe Bild 117 s.',
 'setup.p3.pkg_img': 'Bildgenerator — Stable Diffusion 1.5 DreamShaper LCM auf dem Grafikkern (2,2 GB)',
 'core.kit.img_name': 'Bildgenerator',
}
EN = {
 'setup.pkg.img_name': 'Image generator (text-to-image)',
 'setup.pkg.img_desc': 'Turns text descriptions into images — entirely on the device, no cloud. Stable Diffusion 1.5 (DreamShaper 8 with the LCM sampler) runs on the GPU with compute kernels written specifically for the Mali chip; a 512×512 image in 4 steps takes about 17 seconds. Available over HTTP or in the container via barra-img "…" — as a picture reply alongside AI chat and speech recognition.',
 'setup.pkg.img_fact': '2.2 GB · HTTP service (port 8096) · manual start (imgserver.sh), no boot autostart · holds 1.9 GB of RAM while running',
 'setup.model.img_lcm': 'DreamShaper 8 LCM (Stable Diffusion 1.5)',
 'setup.model.img_lcm_desc': 'Community model DreamShaper 8 (Lykon) on a Stable Diffusion 1.5 base with the LCM sampler baked in: 4 steps instead of 20–30, 2.1 GB (f16) plus the TAESD fast decoder (10 MB). Measured on the node: 512×512 in 16.5–17 s per image with the service running (3.8 s per step; the first image after start takes ~20 s). Photorealistic to illustrative, understands English prompts best; with stock Vulkan the same image took 117 s.',
 'setup.p3.pkg_img': 'Image generator — Stable Diffusion 1.5 DreamShaper LCM on the GPU (2.2 GB)',
 'core.kit.img_name': 'Image generator',
}
ALL_OK = {'de': (', Weckwort wakeserver.sh start.', ', Weckwort wakeserver.sh start, Bildgenerator imgserver.sh start (Port 8096).'),
          'en': None}
BOTH = {'de': (' wakeserver.sh start (Weckwort).', ' wakeserver.sh start (Weckwort), imgserver.sh start (Bildgenerator).'), 'en': None}
for lang, KV in (('de', DE), ('en', EN)):
    for sub in (r'barra-setup\i18n',):
        P = os.path.join(R, sub, lang + '.properties')
        s, bom, crlf = rd(P)
        if 'setup.pkg.img_name=' in s:
            print('i18n schon gepatcht', P); continue
        anchor = 'core.kit.wake_name='
        i = s.find(anchor); assert i > 0, P
        j = s.find('\n', i) + 1
        ins = ''.join('%s=%s\n' % (k, v) for k, v in KV.items())
        s = s[:j] + ins + s[j:]
        # all_ok / both_note ergaenzen (Zeile mit dem Key suchen, Wakeword-Teil erweitern)
        for key, extra in (('core.kit.all_ok=', ('wakeserver.sh start', 'wakeserver.sh start; imgserver.sh start (%s, Port 8096)' % ('Bildgenerator' if lang == 'de' else 'image generator'))),
                           ('core.kit.both_note=', ('wakeserver.sh start', 'wakeserver.sh start; imgserver.sh start (%s)' % ('Bildgenerator' if lang == 'de' else 'image generator')))):
            i = s.find(key); assert i > 0, (P, key)
            j = s.find('\n', i)
            line = s[i:j]
            if 'imgserver' not in line:
                if extra[0] not in line: print('WARN: %s in %s ohne "%s"' % (key, lang, extra[0]))
                else: line = line.replace(extra[0], extra[1], 1)
            s = s[:i] + line + s[j:]
        wr(P, s, bom, crlf); print('ok', P, 'bom=%s' % bom)

# src/i18n = Kopie der Kit-Kataloge (waren seit 23.8. nicht synchron; 4 Alt-Keys setup.p1.pkg_* der entfernten Seite-1-Gruppe entfallen)
import shutil
for lang in ('de','en'):
    shutil.copyfile(os.path.join(R, r'barra-setup\i18n', lang+'.properties'), os.path.join(R, r'src\i18n', lang+'.properties'))
print('src/i18n synchronisiert')
# ---------------- .gitignore ----------------
P = os.path.join(R, '.gitignore')
s, bom, crlf = rd(P)
if 'barra-setup/img-kit/*' not in s:
    s = s.replace('!barra-setup/wake-kit/install-wake.ps1\n', '!barra-setup/wake-kit/install-wake.ps1\nbarra-setup/img-kit/*\n!barra-setup/img-kit/install-img.ps1\n', 1)
    wr(P, s, bom, crlf); print('ok .gitignore')
print('FERTIG')
