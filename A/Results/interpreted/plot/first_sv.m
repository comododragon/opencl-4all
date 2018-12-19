x = 0:31;
labels = [
	"Hotspot"
	"K-Means"
	"LavaMD"
	"NN"
	"NW(1)"
	"NW(2)"

	"SRAD"
	"Backprop(1)"
	"Backprop(2)"
	"LUD(1)"
	"LUD(2)"
	"LUD(3)"
	"Leukocyte(1)"
	"Leukocyte(2)"
	"Hybridsort(1)"
	"Hybridsort(2)"
	"Hybridsort(3)"
	"Hotspot3D"
	"CFD"
	"BPTree"


	"Streamcluster"

	"FFT"

	"MD"
	"MD5Hash"
	"Reduction"
	"SpMV"
	"Stencil2D"
	"Scan"
	"NDRSD(1)"
	"NDRSD(2)"
	"NDRSD(3)"
	"NDRSD(4)"

];
svqdr_sv = [
	0.15
	0.03
	0.07
	0.24
	0.18
	0.19

	0.29
	0.04
	0.21
	0.47
	0.14
	0.01
	0.05
	0.03
	0.15
	0.10
	0.26
	0.02
	0.27
	0.20


	0.03

	0.24

	0.14
	0.30
	0.11
	0.05
	0.12
	0.05
	0.38
	0.35
	0.45
	0.24

];
svqdr_qdr = [
	0.85
	0.97
	0.93
	0.76
	0.82
	0.81

	0.71
	0.96
	0.79
	0.53
	0.86
	0.99
	0.95
	0.97
	0.85
	0.90
	0.74
	0.98
	0.73
	0.80


	0.97

	0.76

	0.86
	0.70
	0.89
	0.95
	0.88
	0.95
	0.62
	0.65
	0.55
	0.76

];
svgtx_sv = [
	0.15
	0.06
	0.05
	0.54
	0.37
	0.38

	0.29
	0.03
	0.28
	0.77
	0.25
	0.01
	0.05
	0.02
	0.14
	0.21
	0.26
	0.01
	0.40
	0.14


	0.02

	0.47

	0.11
	0.21
	0.29
	0.06
	0.21
	0.10
	0.70
	0.68
	0.77
	0.56

];
svgtx_gtx = [
	0.85
	0.94
	0.95
	0.46
	0.63
	0.62

	0.71
	0.97
	0.72
	0.23
	0.75
	0.99
	0.95
	0.98
	0.86
	0.79
	0.74
	0.99
	0.60
	0.86


	0.98

	0.53

	0.89
	0.79
	0.71
	0.94
	0.79
	0.90
	0.30
	0.32
	0.23
	0.44

];

plot(x, svqdr_sv, "-o", x, svgtx_sv, "--x")
axis([0 31 0 1])
view([90 90])
legend("sv x qdr", "sv x gtx")
legend boxoff
set(gca, "xtick", x, "xticklabel", labels, "ylabel", "Relative distance")
set(gcf, "paperunits", "points", "paperposition", [0 0 300 400])
set(gcf, "paperunits", "points", "position", [0 0 300 400])
saveas(gcf, "first_sv.svg", "svg")