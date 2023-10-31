#ifndef CEDAR_INTERFACE_OBJECT_H
#define CEDAR_INTERFACE_OBJECT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	CEDAR_KIND_TOPO   = 0x1,
	CEDAR_KIND_SOLVER = 0x2,
	CEDAR_KIND_MAT    = 0x3,
	CEDAR_KIND_VEC    = 0x4,
	CEDAR_KIND_CONFIG = 0x5
} cedar_kind;

typedef struct {
	int handle;
	int refcount;
	cedar_kind kind;
	void *ptr;
} cedar_object;

#define CEDAR_KIND_MASK  0xf0000000
#define CEDAR_KIND_SHIFT 28
#define CEDAR_KEY_MASK 0x0fffffff
#define CEDAR_GET_KIND(a) ((a & CEDAR_KIND_MASK) >> CEDAR_KIND_SHIFT)
#define CEDAR_SET_KIND(a,kind) ((kind << CEDAR_KIND_SHIFT) | a)

cedar_object *cedar_object_create(cedar_kind kind);
int cedar_object_get(int handle, cedar_object **obj);
int cedar_object_incref(int handle);
int cedar_object_decref(int handle);
int cedar_object_free(int handle);


#ifdef __cplusplus
}
#endif

#endif
