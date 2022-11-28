CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi
CFLAGS += $(shell pkg-config --cflags opencv4)
LDFLAGS += $(shell pkgconf --libs opencv4)

VulkanTest: main.cpp
	glslc  shaders/shader.comp -o shaders/shader.comp.spv
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

.PHONY: test clean

test: VulkanTest
	./VulkanTest Fondo.jpg

clean:
	rm -f VulkanTest
