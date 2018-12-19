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
	"Hotspot3D"

	"SRAD"
	"LUD(1)"
	"LUD(2)"

];
svqdr_sv = [
	0.50
	0.54
	0.05
	0.08
	0.08
	0.32
	0.33
	0.53
	0.07

	0.26
	0.09
	0.10

];
svqdr_qdr = [
	0.50
	0.46
	0.95
	0.92
	0.92
	0.68
	0.67
	0.47
	0.93

	0.74
	0.91
	0.90

];
svgtx_sv = [
	0.52
	0.53
	0.12
	0.21
	0.19
	0.57
	0.57
	0.53
	0.05

	0.26
	0.28
	0.19

];
svgtx_gtx = [
	0.48
	0.47
	0.88
	0.79
	0.81
	0.43
	0.43
	0.47
	0.95

	0.74
	0.72
	0.81

];

plot(x, svqdr_sv, "-o", x, svgtx_sv, "--x")
axis([0 11 0 1])
view([90 90])
legend("sv x qdr", "sv x gtx")
legend boxoff
set(gca, "xtick", x, "xticklabel", labels, "ylabel", "Relative distance")
set(gcf, "paperunits", "points", "paperposition", [0 0 300 400])
set(gcf, "paperunits", "points", "position", [0 0 300 400])
saveas(gcf, "second_sv.svg", "svg")