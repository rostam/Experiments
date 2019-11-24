package main

import "fmt"
type Circle struct {
  x float64
  y float64
  r float64
}

// computation of average of an array
func average(xs []float64) float64 {
  total := 0.0
  for _, v := range xs {
    total += v
  }
  return total / float64(len(xs))
}

// returning multiple variables
func f() (int, int) {
  return 5, 6
}

func first() {
  fmt.Println("1st")
}

func second() {
  fmt.Println("2nd")
}

func main() {
    fmt.Println("Hello World")
    elements := map[string]map[string]string{
    "H": map[string]string{
      "name":"Hydrogen",
      "state":"gas",
    },
    "He": map[string]string{
      "name":"Helium",
      "state":"gas",
    },
    "Li": map[string]string{
      "name":"Lithium",
      "state":"solid",
    },
    "Be": map[string]string{
      "name":"Beryllium",
      "state":"solid",
    },
    "B":  map[string]string{
      "name":"Boron",
      "state":"solid",
    },
    "C":  map[string]string{
      "name":"Carbon",
      "state":"solid",
    },
    "N":  map[string]string{
      "name":"Nitrogen",
      "state":"gas",
    },
    "O":  map[string]string{
      "name":"Oxygen",
      "state":"gas",
    },
    "F":  map[string]string{
      "name":"Fluorine",
      "state":"gas",
    },
    "Ne":  map[string]string{
      "name":"Neon",
      "state":"gas",
    },
  }

  if el, ok := elements["Li"]; ok {
    fmt.Println(el["name"], el["state"])
  }

  xs := []float64{98,93,77,82,83}
  fmt.Println(average(xs))

  x, y := f()
  fmt.Println(x,y)


  add := func(x, y int) int {
    return x + y
  }
  fmt.Println(add(1,1))

  defer second()
  first()

  //This does not go into recover()
//  panic("PANIC")
//  str := recover()
//  fmt.Println(str)

  defer func() {
    str := recover()
    fmt.Println(str)
  }()
  panic("PANIC")//this goes to recover
  

  var c Circle
  fmt.Println(c.x,c.y,c.r)

}
