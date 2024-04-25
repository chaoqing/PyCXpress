# Compiler
CC = c++
PYTHONPATH=../../
LD_PRELOAD:=$(shell find_libpython)

# Compiler flags
CFLAGS = -g -Wall -std=c++17 -fPIC
CFLAGS += $(shell python3-config --cflags --ldflags --embed)
CFLAGS += $(shell PYTHONPATH=$(PYTHONPATH) python3 -m PyCXpress --includes)

# The build target executable
TARGET = example.out

# source files
SOURCES = main.cpp

# object files
OBJECTS = $(SOURCES:%.cpp=%.o)
DEPENDS = $(SOURCES:%.cpp=.%.d)

all: $(TARGET)

-include $(DEPENDS)

$(TARGET): $(OBJECTS) Makefile
	$(CC) -o $(TARGET) $(OBJECTS) $(CFLAGS)

.%.d: %.cpp Makefile
	$(CC) -c $< -MM -MP -o $@ $(CFLAGS)

%.o: %.cpp Makefile
	$(CC) -c $< -o $@ $(CFLAGS)

.PHONY: clean run memcheck

# Clean target
clean:
	rm -f $(OBJECTS) $(DEPENDS) $(TARGET)

run: $(TARGET)
	$(info [reminding]: use LD_PRELOAD to load libpython before numpy import)
	PYTHONPATH=$(PYTHONPATH) LD_PRELOAD=$(LD_PRELOAD) ./$(TARGET)

memcheck: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes -s ./$(TARGET)
