package cuvs

import (
	"os/exec"
	"testing"
)

func CheckGpuMemory() error {
	// run nvidia-smi
	cmd := exec.Command("nvidia-smi")
	out, err := cmd.Output()
	println("nvidia-smi output (CheckGpuMemory()): ", string(out))
	if err != nil {
		return err
	}
	return nil
}

func TestMemoryResource(t *testing.T) {
	err := EnablePoolMemoryResource(50, 100, false)
	if err != nil {
		t.Error("failed to enable pool memory resource")
	}
	CheckGpuMemory()

	err = ResetMemoryResource()

	if err != nil {
		t.Error("failed to reset memory resource")
	}

	CheckGpuMemory()

}
