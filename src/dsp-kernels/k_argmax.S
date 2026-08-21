# k_argmax — greedy Token-Auswahl auf dem Callisto-DSP (Basis-ISA, kein FP noetig).
# Eingabe : buf[0]=N, buf[1..N]=float32-Bitmuster der Logits (int32)
# Ausgabe : buf[0]=argmax-Index, buf[1]=monotoner Schluessel des Maximums
# Trick   : IEEE-float -> monoton signed int: key = v ^ ((v>>31 arith) >>> 1)
#           (negative Floats: v ^ 0x7fffffff), dann signierter Vergleich (bge).
	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base
	l32i	a4, a3, 0	# N
	addi	a6, a3, 4	# ptr = &buf[1]
	movi	a5, 0		# i
	movi	a8, 0		# best index
	movi	a9, 1
	slli	a9, a9, 31	# best key = INT_MIN
.Lloop:
	beqz	a4, .Ldone
	l32i	a7, a6, 0
	srai	a10, a7, 31
	srli	a10, a10, 1
	xor	a7, a7, a10	# key
	bge	a9, a7, .Lskip	# best >= key -> skip
	mov	a9, a7
	mov	a8, a5
.Lskip:
	addi	a6, a6, 4
	addi	a5, a5, 1
	addi	a4, a4, -1
	j	.Lloop
.Ldone:
	s32i	a8, a3, 0
	s32i	a9, a3, 4
	memw
	movi	a2, 0
	retw.n
