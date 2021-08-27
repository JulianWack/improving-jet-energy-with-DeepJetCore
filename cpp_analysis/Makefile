

#
# This file might need some adjustments but should serve as a good basis
#

PYTHON_INCLUDE = `python-config --includes`
PYTHON_LIB=`python-config --libs`

ROOTSTUFF=`root-config --libs --glibs --ldflags`
ROOTCFLAGS=`root-config  --cflags`

CPP_FILES := $(wildcard src/*.cpp)
OBJ_FILES := $(addprefix obj/,$(notdir $(CPP_FILES:.cpp=.o)))
LD_FLAGS := `root-config --cflags --glibs`  -lMathMore -L${DEEPJETCORE}/compiled -ldeepjetcorehelpers -lquicklz
CC_FLAGS := -fPIC -g -Wall `root-config --cflags`
CC_FLAGS += -I./interface -I${DEEPJETCORE}/compiled/interface
#CC_FLAGS += -MMD


all: $(patsubst bin/%.cpp, %, $(wildcard bin/*.cpp))


%: bin/%.cpp  $(OBJ_FILES) 
	g++ $(CC_FLAGS) $(LD_FLAGS) $(OBJ_FILES) $< -o $@ 


obj/%.o: src/%.cpp
	g++ $(CC_FLAGS) -c -o $@ $<


clean: 
	rm -f obj/*.o obj/*.d
	rm -f %
