	.text
	.global kfn
# Software IEEE-754 single-precision multiply, integer ops only.
# out[2] = in[0] * in[1]   (normal numbers + signed zero; no inf/nan/subnormal)
kfn:
	entry	a1, 48
	l32i	a3, a3, 0
	l32i	a3, a3, 0
	l32i	a3, a3, 8	# base
	l32i	a4, a3, 0	# A bits
	l32i	a5, a3, 4	# B bits
	xor	a6, a4, a5
	extui	a6, a6, 31, 1	# sign
	extui	a8, a4, 23, 8	# eA
	extui	a9, a5, 23, 8	# eB
	beqz	a8, zero
	beqz	a9, zero
	slli	a10, a4, 9
	srli	a10, a10, 9	# mA[22:0]
	slli	a11, a5, 9
	srli	a11, a11, 9	# mB[22:0]
	movi	a7, 1
	slli	a7, a7, 23	# 0x800000
	or	a10, a10, a7	# mA | implicit1
	or	a11, a11, a7	# mB | implicit1
	mull	a12, a10, a11	# product lo
	muluh	a13, a10, a11	# product hi
	add	a8, a8, a9
	addi	a8, a8, -127	# e = eA+eB-127
	extui	a15, a13, 15, 1	# bit47 of product
	beqz	a15, caseB
	slli	a14, a13, 8
	srli	a7, a12, 24
	or	a14, a14, a7	# m24 = P>>24
	addi	a8, a8, 1
	extui	a9, a12, 23, 1	# guard bit
	j	rnd
caseB:
	slli	a14, a13, 9
	srli	a7, a12, 23
	or	a14, a14, a7	# m24 = P>>23
	extui	a9, a12, 22, 1	# guard bit
rnd:
	add	a14, a14, a9	# round to nearest
	extui	a15, a14, 24, 1	# rounding carry?
	beqz	a15, asmb
	srli	a14, a14, 1
	addi	a8, a8, 1
asmb:
	movi	a7, 1
	slli	a7, a7, 23
	addi	a7, a7, -1	# 0x7FFFFF
	and	a14, a14, a7	# drop implicit bit
	slli	a8, a8, 23
	or	a14, a14, a8	# | exp<<23
	slli	a6, a6, 31
	or	a14, a14, a6	# | sign<<31
	j	store
zero:
	slli	a14, a6, 31	# signed zero
store:
	s32i	a14, a3, 8	# out[2] = result
	memw
	movi	a2, 0
	retw.n
