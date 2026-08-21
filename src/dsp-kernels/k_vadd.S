	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base ptr to int32 array
	# read all: a[0..3] @0..12 , b[0..3] @16..28
	l32i	a4, a3, 0
	l32i	a5, a3, 4
	l32i	a6, a3, 8
	l32i	a7, a3, 12
	l32i	a8, a3, 16
	l32i	a9, a3, 20
	l32i	a10, a3, 24
	l32i	a11, a3, 28
	add	a4, a4, a8
	add	a5, a5, a9
	add	a6, a6, a10
	add	a7, a7, a11
	# write all
	s32i	a4, a3, 0
	s32i	a5, a3, 4
	s32i	a6, a3, 8
	s32i	a7, a3, 12
	memw
	movi	a2, 0
	retw.n
