package main

import (
	"github.com/mitroadmaps/gomapinfer/common"

	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

func dfs(node *common.Node, seen map[int]bool) {
	if seen[node.ID] {
		return
	}
	seen[node.ID] = true
	for _, edge := range node.Out {
		dfs(edge.Dst, seen)
	}
}

func getConnectedComponents(fname string, isgt bool) []common.Rectangle {
	g, err := common.ReadGraph(fname)
	if err != nil {
		panic(err)
	}
	seen := make(map[int]bool)
	var rects []common.Rectangle
	for _, node := range g.Nodes {
		if seen[node.ID] {
			continue
		}
		component := make(map[int]bool)
		dfs(node, component)
		if !isgt && len(component) < 6 {
			continue
		}
		rect := common.EmptyRectangle
		for nodeID := range component {
			seen[nodeID] = true
			rect = rect.Extend(g.Nodes[nodeID].Point)
		}
		rects = append(rects, rect)
	}
	return rects
}

func main() {
	gtFname := os.Args[1]
	inferredFname := os.Args[2]
	unmatchedPath := os.Args[3]

	var ignoreRects []common.Rectangle
	files, err := ioutil.ReadDir(unmatchedPath)
	if err != nil {
		panic(err)
	}
	for _, fi := range files {
		if !strings.HasSuffix(fi.Name(), ".json") {
			continue
		}
		var jsonData []int
		bytes, err := ioutil.ReadFile(unmatchedPath + fi.Name())
		if err != nil {
			panic(err)
		}
		if err := json.Unmarshal(bytes, &jsonData); err != nil {
			panic(err)
		}
		ignoreRects = append(ignoreRects, common.Rectangle{
			Min: common.Point{float64(jsonData[0]), float64(jsonData[1])},
			Max: common.Point{float64(jsonData[2]), float64(jsonData[3])},
		})
	}

	rects1 := getConnectedComponents(gtFname, true)
	rects2 := getConnectedComponents(inferredFname, false)

	matched1 := make(map[int]bool)
	matched2 := make(map[int]bool)
	for idx1, rect1 := range rects1 {
		bestCandidateIdx := -1
		var bestIntersectArea float64
		for idx2, rect2 := range rects2 {
			area := rect1.Intersection(rect2).Area()
			if (matched2[idx2] && false) || area <= 0 {
				continue
			}

			// no one-to-one
			matched1[idx1] = true
			matched2[idx2] = true

			if bestCandidateIdx == -1 || area > bestIntersectArea {
				bestCandidateIdx = idx2
				bestIntersectArea = area
			}
		}
		if bestCandidateIdx != -1 {
			matched1[idx1] = true
			matched2[bestCandidateIdx] = true
		}
	}

	var countNonIgnoreRects2 int
	ignoredRects := make(map[int]bool)
	for idx2, rect2 := range rects2 {
		ignore := false
		for _, chk := range ignoreRects {
			if rect2.Intersection(chk).Area() > 0 {
				ignore = true
				break
			}
		}
		if !ignore || matched2[idx2] {
			countNonIgnoreRects2++
		} else if ignore {
			ignoredRects[idx2] = true
		}
	}

	recall := float64(len(matched1))/float64(len(rects1))
	precision := float64(len(matched2))/float64(countNonIgnoreRects2)
	fmt.Printf("gt (%d of %d) inferred (%d of %d->%d)\n", len(matched1), len(rects1), len(matched2), len(rects2), countNonIgnoreRects2)
	fmt.Printf("%v %v\n", precision, recall)
}
