x = 0:19;
labels = [
	"Hotspot"
	"K-Means"
	"LavaMD"




	"SRAD"

	"Backprop(2)"



	"Leukocyte(1)"
	"Leukocyte(2)"

	"Hybridsort(2)"
	"Hybridsort(3)"
	"Hotspot3D"
	"CFD"
	"BPTree"




	"FFT"
	"GEMM"

	"MD5Hash"

	"SpMV"



	"NDRSD(2)"
	"NDRSD(3)"
	"NDRSD(4)"
	"NDRSDFull"
];
a10qdr_a10 = [
	0.10
	0.01
	0.05




	0.04

	0.09



	0.00
	0.00

	0.05
	0.05
	0.02
	0.24
	0.08




	0.21
	0.19

	0.24

	0.04



	0.04
	0.34
	0.08
	0.33
];
a10qdr_qdr = [
	0.90
	0.99
	0.95




	0.96

	0.91



	1.00
	1.00

	0.95
	0.95
	0.98
	0.76
	0.92




	0.79
	0.81

	0.76

	0.96



	0.96
	0.66
	0.92
	0.67
];
a10gtx_a10 = [
	0.10
	0.03
	0.03




	0.04

	0.12



	0.00
	0.00

	0.11
	0.05
	0.01
	0.36
	0.05




	0.43
	0.48

	0.16

	0.05



	0.13
	0.67
	0.25
	0.65
];
a10gtx_gtx = [
	0.90
	0.97
	0.97




	0.96

	0.88



	1.00
	1.00

	0.89
	0.95
	0.99
	0.64
	0.95




	0.57
	0.52

	0.84

	0.95



	0.87
	0.33
	0.75
	0.35
];

plot(x, a10qdr_a10, "-o", x, a10gtx_a10, "--x")
axis([0 19 0 1])
view([90 90])
legend("a10 x qdr", "a10 x gtx")
legend boxoff
set(gca, "xtick", x, "xticklabel", labels, "ylabel", "Relative distance")
set(gcf, "paperunits", "points", "paperposition", [0 0 300 400])
set(gcf, "paperunits", "points", "position", [0 0 300 400])
saveas(gcf, "first_a10.svg", "svg")