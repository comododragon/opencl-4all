x = 0:2;
labels = [
	"NW"
	"Hotspot"
	"Hotspot3D"

];
svqdr_sv = [
	0.83
	0.20
	0.32

];
svqdr_qdr = [
	0.17
	0.80
	0.68

];
svgtx_sv = [
	0.93
	0.19
	0.25

];
svgtx_gtx = [
	0.07
	0.81
	0.75

];

plot(x, svqdr_sv, "-o", x, svgtx_sv, "--x")
axis([0 2 0 1])
view([90 90])
legend("sv x qdr", "sv x gtx")
legend boxoff
set(gca, "xtick", x, "xticklabel", labels, "ylabel", "Relative distance")
set(gcf, "paperunits", "points", "paperposition", [0 0 300 400])
set(gcf, "paperunits", "points", "position", [0 0 300 400])
saveas(gcf, "third_sv.svg", "svg")