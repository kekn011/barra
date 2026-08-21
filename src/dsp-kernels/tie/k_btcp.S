	.text
	.global kfn
kfn:
	entry	a1, 48
	movi	a5, 0xff
	wsr.cpenable	a5
	rsync
	s32i	a0, a1, 0
	s32i	a3, a1, 4
	.byte	0x9f,0x6b,0x85,0x00,0x08,0x19,0x55,0x24
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
