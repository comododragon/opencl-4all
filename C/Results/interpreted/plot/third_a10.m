x = 0:1;
labels = [

	"Hotspot"
	"Hotspot3D"

];
a10qdr_a10 = [

	0.14
	0.26

];
a10qdr_qdr = [

	0.86
	0.74

];
a10gtx_a10 = [

	0.14
	0.20

];
a10gtx_gtx = [

	0.86
	0.80

];

plot(x, a10qdr_a10, "-o", x, a10gtx_a10, "--x")
axis([0 1 0 1])
view([90 90])
legend("a10 x qdr", "a10 x gtx")
legend boxoff
set(gca, "xtick", x, "xticklabel", labels, "ylabel", "Relative distance")
set(gcf, "paperunits", "points", "paperposition", [0 0 300 400])
set(gcf, "paperunits", "points", "position", [0 0 300 400])
saveas(gcf, "third_a10.svg", "svg")