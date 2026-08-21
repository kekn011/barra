	.text
	.global kfn
kfn:
	entry	a1, 32
	# a2 = argc, a3 = argv (pointer to array of arg-descriptor pointers)
	l32i	a8, a3, 0	# argv[0]
	l32i	a9, a3, 4	# argv[1] (raw, no deref)
	l32i	a10, a8, 0	# *(argv[0])
	l32i	a10, a10, 8	# base = *(*(argv[0])+8)  (buffer data)
	# all writes now
	s32i	a2, a10, 0	# buf[0] = argc
	s32i	a8, a10, 4	# buf[1] = argv[0]
	s32i	a9, a10, 8	# buf[2] = argv[1]
	memw
	movi	a2, 0
	retw.n
