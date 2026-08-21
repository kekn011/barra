	.text
	.global kfn
# Batch-vscale: buf[0]=N (Anzahl), buf[1]=Skalar s, buf[2..N+1] *= s (in-place).
# Ein Dispatch verarbeitet N Elemente -> amortisiert den ~ms Command-Overhead.
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# a3 = base (Buffer-Datenzeiger)
	l32i	a4, a3, 0	# a4 = N   = buf[0]
	l32i	a5, a3, 4	# a5 = s   = buf[1]
	addi	a6, a3, 8	# a6 = &buf[2]
.Lloop:
	beqz	a4, .Ldone
	l32i	a7, a6, 0
	mull	a7, a7, a5
	s32i	a7, a6, 0
	addi	a6, a6, 4
	addi	a4, a4, -1
	j	.Lloop
.Ldone:
	memw
	movi	a2, 0
	retw.n
