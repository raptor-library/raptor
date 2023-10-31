https://github.com/cedar-framework/cedar/blob/master/include/cedar/interface/object.h

C++ struct that stores everything we need for a matrix
 - store and address to this in the void *ptr of cedar_object

leave this (nless there are more than 16):
#define CEDAR_KIND_MASK  0xf0000000
#define CEDAR_KIND_SHIFT 28
#define CEDAR_KEY_MASK 0x0fffffff
#define CEDAR_GET_KIND(a) ((a & CEDAR_KIND_MASK) >> CEDAR_KIND_SHIFT)
#define CEDAR_SET_KIND(a,kind) ((kind << CEDAR_KIND_SHIFT) | a)

interface to generic objects:
cedar_object *cedar_object_create(cedar_kind kind);
int cedar_object_get(int handle, cedar_object **obj);
int cedar_object_incref(int handle);
int cedar_object_decref(int handle);
int cedar_object_free(int handle);

source: https://github.com/cedar-framework/cedar/blob/master/src/interface/object.cc

all objects are stored here:
https://github.com/cedar-framework/cedar/blob/master/src/interface/object.cc#L7

replace cedar with raptor

declarations:
https://github.com/cedar-framework/cedar/blob/master/include/cedar/capi.h
source:
https://github.com/cedar-framework/cedar/blob/master/src/2d/interface/c/vec.cc
