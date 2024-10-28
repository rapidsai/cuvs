package cuvs

import (
	"os/exec"
	"testing"
)

func CheckGpuMemory() error {
	cmd := exec.Command("nvidia-smi")
	out, err := cmd.Output()
	println("nvidia-smi output (CheckGpuMemory()): ", string(out))
	return err
}

func TestMemoryResource(t *testing.T) {
	t.Log("Starting memory resource test")

	mem := NewCuvsPoolMemory(60, 100, false)
	defer mem.Close()

	t.Log("Instantiating memory pool...")
	mem.Instantiate()

	if err := CheckGpuMemory(); err != nil {
		t.Fatal("GPU memory check failed after instantiation:", err)
	}

	t.Log("Releasing memory pool...")
	mem.Release()

	if err := CheckGpuMemory(); err != nil {
		t.Fatal("GPU memory check failed after release:", err)
	}

	t.Log("Memory resource test completed")
}
