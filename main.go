package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"io/ioutil"
	"log"
	"math"
	"os"
	"reflect"
	"sort"
	"time"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

const (
	height = 64
	width  = 64
)

var emotionTable = []string{
	"neutral",
	"happiness",
	"surprise",
	"sadness",
	"anger",
	"disgust",
	"fear",
	"contempt",
}

func main() {
	model := flag.String("model", "model/model.onnx", "path to the model file")
	input := flag.String("input", "images/avatar64.png", "path to the input file")
	h := flag.Bool("h", false, "help")
	flag.Parse()
	if *h {
		flag.Usage()
		os.Exit(0)
	}
	if _, err := os.Stat(*model); err != nil && os.IsNotExist(err) {
		log.Fatalf("%v does not exist", *model)
	}
	if _, err := os.Stat(*input); err != nil && *input != "-" && os.IsNotExist(err) {
		log.Fatalf("%v does not exist", *input)
	}

	//
	// Step 1: Create the execution backend and load the model
	//
	backend := gorgonnx.NewGraph()
	m := onnx.NewModel(backend)

	// read the onnx model
	b, err := ioutil.ReadFile(*model)
	if err != nil {
		log.Fatal(err)
	}
	// Decode it into the model
	err = m.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	//
	// Step 2: Create model inputs
	//
	// The number of inputs depends on the model.

	inputStream, err := os.Open(*input)
	if err != nil {
		log.Fatal(err)
	}

	img, err := png.Decode(inputStream)
	if err != nil {
		log.Fatal(err)
	}

	imgGray, ok := img.(*image.Gray)
	if !ok {
		log.Fatal("Please give a gray image as input")
	}
	inputT := tensor.New(tensor.WithShape(1, 1, height, width), tensor.Of(tensor.Float32))
	err = GrayToBCHW(imgGray, inputT)
	if err != nil {
		log.Fatal(err)
	}
	m.SetInput(0, inputT)

	//
	// Step 3: Run the model
	//
	start := time.Now()
	err = backend.Run()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Computation time: %v\n", time.Since(start))
	computedOutputT, err := m.GetOutputTensors()
	if err != nil {
		log.Fatal(err)
	}

	//
	// Step 4: Gather results
	//
	result := classify(softmax(computedOutputT[0].Data().([]float32)))
	fmt.Printf("%v / %2.2f%%\n", result[0].emotion, result[0].weight*100)
	fmt.Printf("%v / %2.2f%%\n", result[1].emotion, result[1].weight*100)
}

type testingT struct{}

func (t *testingT) Errorf(format string, args ...interface{}) {
	log.Fatalf(format, args...)
}

func softmax(input []float32) []float32 {
	var sumExp float64
	output := make([]float32, len(input))
	for i := 0; i < len(input); i++ {
		sumExp += math.Exp(float64(input[i]))
	}
	for i := 0; i < len(input); i++ {
		output[i] = float32(math.Exp(float64(input[i]))) / float32(sumExp)
	}
	return output
}

func classify(input []float32) emotions {
	result := make(emotions, len(input))
	for i := 0; i < len(input); i++ {
		result[i] = emotion{
			emotion: emotionTable[i],
			weight:  input[i],
		}
	}
	sort.Sort(sort.Reverse(result))
	return result
}

func createInputStream(input *string) io.Reader {
	var inputStream io.Reader
	if *input != "-" {
		imgContent, err := os.Open(*input)
		if err != nil {
			log.Fatal(err)
		}
		defer imgContent.Close()
		inputStream = imgContent
	} else {
		inputStream = os.Stdin
	}
	return inputStream
}

type emotions []emotion
type emotion struct {
	emotion string
	weight  float32
}

func (e emotions) Len() int           { return len(e) }
func (e emotions) Swap(i, j int)      { e[i], e[j] = e[j], e[i] }
func (e emotions) Less(i, j int) bool { return e[i].weight < e[j].weight }

// GrayToBCHW convert an image to a BCHW tensor
// this function returns an error if:
//
//   - dst is not a pointer
//   - dst's shape is not 4
//   - dst' second dimension is not 1
//   - dst's third dimension != i.Bounds().Dy()
//   - dst's fourth dimension != i.Bounds().Dx()
//   - dst's type is not float32 or float64 (temporary)
func GrayToBCHW(img *image.Gray, dst tensor.Tensor) error {
	w := img.Bounds().Dx()
	h := img.Bounds().Dy()
	err := verifyBCHWTensor(dst, h, w, true)
	if err != nil {
		return err
	}

	switch dst.Dtype() {
	case tensor.Float32:
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				color := img.GrayAt(x, y)
				err := dst.SetAt(float32(color.Y), 0, 0, y, x)
				if err != nil {
					return err
				}
			}
		}
	case tensor.Float64:
		for x := 0; x < w; x++ {
			for y := 0; y < h; y++ {
				color := img.GrayAt(x, y)
				err := dst.SetAt(float64(color.Y), x, y)
				if err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("%v not handled yet", dst.Dtype())
	}
	return nil
}

func verifyBCHWTensor(dst tensor.Tensor, h, w int, cowardMode bool) error {
	// check if tensor is a pointer
	rv := reflect.ValueOf(dst)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("cannot decode image into a non pointer or a nil receiver")
	}
	// check if tensor is compatible with BCHW (4 dimensions)
	if len(dst.Shape()) != 4 {
		return fmt.Errorf("Expected a 4 dimension tensor, but receiver has only %v", len(dst.Shape()))
	}
	// Check the batch size
	if dst.Shape()[0] != 1 {
		return errors.New("only batch size of one is supported")
	}
	if cowardMode && dst.Shape()[1] != 1 {
		return errors.New("Cowardly refusing to insert a gray scale into a tensor with more than one channel")
	}
	if dst.Shape()[2] != h || dst.Shape()[3] != w {
		return fmt.Errorf("cannot fit image into tensor; image is %v*%v but tensor is %v*%v", h, w, dst.Shape()[2], dst.Shape()[3])
	}
	return nil
}
