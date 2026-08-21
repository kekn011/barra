	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base
	# A(2x2)@0..12, B(2x2)@16..28, C(2x2)@32..44
	l32i	a4, a3, 0	# A0
	l32i	a5, a3, 4	# A1
	l32i	a6, a3, 8	# A2
	l32i	a7, a3, 12	# A3
	l32i	a8, a3, 16	# B0
	l32i	a9, a3, 20	# B1
	l32i	a10, a3, 24	# B2
	l32i	a11, a3, 28	# B3
	mull	a12, a4, a8
	mull	a2, a5, a10
	add	a12, a12, a2	# C0 = A0*B0 + A1*B2
	mull	a13, a4, a9
	mull	a2, a5, a11
	add	a13, a13, a2	# C1 = A0*B1 + A1*B3
	mull	a14, a6, a8
	mull	a2, a7, a10
	add	a14, a14, a2	# C2 = A2*B0 + A3*B2
	mull	a15, a6, a9
	mull	a2, a7, a11
	add	a15, a15, a2	# C3 = A2*B1 + A3*B3
	s32i	a12, a3, 32
	s32i	a13, a3, 36
	s32i	a14, a3, 40
	s32i	a15, a3, 44
	memw
	movi	a2, 0
	retw.n
