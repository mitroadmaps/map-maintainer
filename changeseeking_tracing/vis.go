package main

import (
	"github.com/mitroadmaps/gomapinfer/common"
	"github.com/mitroadmaps/gomapinfer/image"

	"fmt"
	"os"
	"strconv"
)

func vis(basemap *common.Graph, inferred *common.Graph, x int, y int) {
	outFname := fmt.Sprintf("%d_%d.jpg", x, y)
	satFname := fmt.Sprintf("imagery-new/mass_%d_%d_sat.jpg", x, y)

	im := image.ReadImage(satFname)
	tilePoint := common.Point{float64(x), float64(y)}.Scale(4096)
	tileRect := common.Rectangle{tilePoint, tilePoint.Add(common.Point{4096, 4096})}

	drawGraph := func(g *common.Graph, color [3]uint8, width int) {
		for _, edge := range g.Edges {
			if !tileRect.Contains(edge.Src.Point) && !tileRect.Contains(edge.Dst.Point) {
				continue
			}
			start := edge.Src.Point.Sub(tilePoint)
			end := edge.Dst.Point.Sub(tilePoint)
			for _, p := range common.DrawLineOnCells(int(start.X), int(start.Y), int(end.X), int(end.Y), 4096, 4096) {
				image.DrawRect(im, p[0], p[1], width, color)
			}
		}
	}

	drawGraph(inferred, [3]uint8{255, 0, 0}, 2)
	drawGraph(basemap, [3]uint8{255, 0, 255}, 1)
	image.WriteImage(outFname, im)
}

func main() {
	baseFname := os.Args[1]
	fname := os.Args[2]
	x, _ := strconv.Atoi(os.Args[2])
	y, _ := strconv.Atoi(os.Args[3])

	basemap, err := common.ReadGraph(baseFname)
	if err != nil {
		panic(err)
	}
	inferred, err := common.ReadGraph(fname)
	if err != nil {
		panic(err)
	}

	vis(basemap, inferred, x, y)
}
