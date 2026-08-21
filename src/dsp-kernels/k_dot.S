	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base
	# a[0..3]@0..12, b[0..3]@16..28 ; dot = sum(a[i]*b[i]) -> buf[0]
	l32i	a4, a3, 0
	l32i	a5, a3, 4
	l32i	a6, a3, 8
	l32i	a7, a3, 12
	l32i	a8, a3, 16
	l32i	a9, a3, 20
	l32i	a10, a3, 24
	l32i	a11, a3, 28
	mull	a4, a4, a8
	mull	a5, a5, a9
	mull	a6, a6, a10
	mull	a7, a7, a11
	add	a4, a4, a5
	add	a6, a6, a7
	add	a4, a4, a6
	s32i	a4, a3, 0
	memw
	movi	a2, 0
	retw.n
