	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# a3 = device data pointer
	l32i	a7, a3, 0	# k = buf[0]
	l32i	a4, a3, 4
	l32i	a5, a3, 8
	l32i	a6, a3, 12
	mull	a4, a4, a7
	mull	a5, a5, a7
	mull	a6, a6, a7
	s32i	a4, a3, 4
	s32i	a5, a3, 8
	s32i	a6, a3, 12
	memw
	movi	a2, 0
	retw.n
