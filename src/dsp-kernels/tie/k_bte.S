	.text
	.global kfn
kfn:
	entry	a1, 48
	s32i	a0, a1, 0
	s32i	a3, a1, 4
	.byte	0x5e,0xb0,0x41,0x0e,0xe0,0xc6,0xc4,0x05
	l32i	a0, a1, 0
	l32i	a3, a1, 4
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8
	movi	a4, 291
	s32i	a4, a3, 0
	memw
	movi	a2, 0
	retw.n
