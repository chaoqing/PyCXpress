# Compiler
CC = c++

# Compiler flags
CFLAGS = -g -Wall -std=c++17 -fPIC
CFLAGS = -I../src/cpp
CFLAGS += $(shell python3-config --cflags --ldflags --embed)
CFLAGS += $(shell PYTHONPATH=../src/ python3 -m PyCXpress --includes) -I ../src/include

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
	PYTHONPATH=../src/ ./$(TARGET)

memcheck: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes -s ./$(TARGET)