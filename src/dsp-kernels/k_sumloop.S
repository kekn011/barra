	.text
	.global kfn
kfn:
	entry	a1, 32
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base
	l32i	a4, a3, 0	# N = buf[0]
	movi	a5, 0		# sum = 0
	addi	a6, a3, 4	# ptr = &buf[1]
.Lloop:
	beqz	a4, .Ldone
	l32i	a7, a6, 0
	add	a5, a5, a7
	addi	a6, a6, 4
	addi	a4, a4, -1
	j	.Lloop
.Ldone:
	s32i	a5, a3, 0	# buf[0] = sum(buf[1..N])
	memw
	movi	a2, 0
	retw.n
