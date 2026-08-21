	.text
	.global kfn
kfn:
	entry	a1, 32
	movi	a5, 255
	wsr.cpenable a5
	rsync
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base ptr to float array
	lsi	f0, a3, 0	# f0 = in[0]
	lsi	f1, a3, 4
	lsi	f2, a3, 8
	lsi	f3, a3, 12
	mul.s	f1, f1, f0
	mul.s	f2, f2, f0
	mul.s	f3, f3, f0
	ssi	f1, a3, 4
	ssi	f2, a3, 8
	ssi	f3, a3, 12
	memw
	movi	a2, 0
	retw.n
