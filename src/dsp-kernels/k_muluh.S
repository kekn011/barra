	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base
	l32i	a4, a3, 0	# a = in[0]
	l32i	a5, a3, 4	# b = in[1]
	mull	a6, a4, a5	# low 32
	muluh	a7, a4, a5	# high 32 (unsigned)
	s32i	a6, a3, 0	# out[0] = low
	s32i	a7, a3, 4	# out[1] = high
	memw
	movi	a2, 0
	retw.n
