	.text
	.global kfn
kfn:
	entry	a1, 48
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8
	movi	a4, 291
	s32i	a4, a3, 0
	memw
	movi	a2, 0
	retw.n
