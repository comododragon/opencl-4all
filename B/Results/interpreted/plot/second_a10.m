x = 0:11;
labels = [
	"Poisson(1)"
	"Poisson(2)"
	"Bitonic(1)"
	"Bitonic(2)"
	"Bitonic(3)"
	"NW(1)"
	"NW(2)"
	"Hotspot"


	"SRAD"
	"LUD(1)"
	"LUD(2)"
	"LUD(3)"
];
a10qdr_a10 = [
	0.53
	0.41
	0.03
	0.06
	0.03
	0.12
	0.13
	0.43


	0.06
	0.07
	0.04
	0.26
];
a10qdr_qdr = [
	0.47
	0.59
	0.97
	0.94
	0.97
	0.88
	0.87
	0.57


	0.94
	0.93
	0.96
	0.74
];
a10gtx_a10 = [
	0.55
	0.41
	0.08
	0.17
	0.09
	0.28
	0.28
	0.43


	0.06
	0.23
	0.09
	0.25
];
a10gtx_gtx = [
	0.45
	0.59
	0.92
	0.83
	0.91
	0.72
	0.72
	0.57


	0.94
	0.77
	0.91
	0.75
];

plot(x, a10qdr_a10, "-o", x, a10gtx_a10, "--x")
axis([0 11 0 1])
view([90 90])
legend("a10 x qdr", "a10 x gtx")
legend boxoff
set(gca, "xtick", x, "xticklabel", labels, "ylabel", "Relative distance")
set(gcf, "paperunits", "points", "paperposition", [0 0 300 400])
set(gcf, "paperunits", "points", "position", [0 0 300 400])
saveas(gcf, "second_a10.svg", "svg")