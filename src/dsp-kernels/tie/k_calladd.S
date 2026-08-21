	.text
	.global kfn
# Forward (argc=a2, argv=a3) to reference fn add_integer_array @ vaddr 0x228
# (full of xt-clang FLIX/TIE bundles) in the same loaded image. Load-base
# independent: read runtime PC via call0, add compile-time delta.
# Fixed 3-byte encodings => obj_off(label 1)=0x0f; delta=0x228-0x14c-0x0f=0xcd.
kfn:
	entry	a1, 48
	or	a6, a0, a0	# save return PC
	or	a10, a2, a2	# argc -> callee a2
	or	a11, a3, a3	# argv -> callee a3
	call0	1f
1:
	movi	a9, 0xd0
	add	a8, a0, a9	# a8 = runtime &add_integer_array
	callx8	a8
	or	a0, a6, a6	# restore return PC
	memw
	movi	a2, 0
	retw.n
