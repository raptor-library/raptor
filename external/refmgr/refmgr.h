#pragma once

#include <boost/preprocessor/variadic/to_seq.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#ifdef __cplusplus
#include <vector>
#include <cstdlib>
extern "C" {
#endif

#define REFMGR_SUCCESS     0
#define REFMGR_ERR_KIND    1
#define REFMGR_ERR_HANDLE  2
#define REFMGR_ERR_OTHER   3

#define REFMGR_KIND_MASK  0xf0000000
#define REFMGR_KIND_SHIFT 28
#define REFMGR_KEY_MASK 0x0fffffff
#define REFMGR_GET_KIND(a) ((a & REFMGR_KIND_MASK) >> REFMGR_KIND_SHIFT)
#define REFMGR_SET_KIND(a,kind) ((kind << REFMGR_KIND_SHIFT) | a)

#define REFMGR_ENUMCAT(r, data, elem) BOOST_PP_CAT(elem, data)=r,
#define REFMGR_DEFTYPE(r, data, elem) typedef data elem;
#define REFMGR_DEFNULL(r, data, elem) static int BOOST_PP_CAT(elem, data) = BOOST_PP_CAT(0x, BOOST_PP_CAT(r, 0000000));
#define REFMGR_DEFOBJ(libname) typedef struct { int handle; int refcount; BOOST_PP_CAT(libname, _kind) kind; void *ptr; } BOOST_PP_CAT(libname, _object);
#define REFMGR_DECFUN(libname) int BOOST_PP_CAT(libname, _object_get)(int handle, BOOST_PP_CAT(libname, _object) **obj); \
	int BOOST_PP_CAT(libname, _object_incref)(int handle); \
	int BOOST_PP_CAT(libname, _object_decref)(int handle); \
	int BOOST_PP_CAT(libname, _object_free)(int handle);
#define REFMGR_INIT_SEQ(libname, seq) typedef enum {BOOST_PP_SEQ_FOR_EACH(REFMGR_ENUMCAT, _KIND, seq) REFMGR_KIND_BOUND} BOOST_PP_CAT(libname, _kind); \
	BOOST_PP_SEQ_FOR_EACH(REFMGR_DEFTYPE, int, seq) \
		BOOST_PP_SEQ_FOR_EACH(REFMGR_DEFNULL, _NULL, seq) \
		REFMGR_DEFOBJ(libname) \
		REFMGR_DECFUN(libname)
#define REFMGR_INIT(libname, ...) REFMGR_INIT_SEQ(libname, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
#define REFMGR_GEN_HEADER(libname) \
	static std::vector<BOOST_PP_CAT(libname, _object) *> objects; \
	extern "C" {
#define REFMGR_OBJTYPE(libname) BOOST_PP_CAT(libname, _object)
#define REFMGR_KINDTYPE(libname) BOOST_PP_CAT(libname, _kind)
#define REFMGR_GEN_OBJCREATE(lib)                                              \
	REFMGR_OBJTYPE(lib) *                                                      \
	BOOST_PP_CAT(lib, _object_create)(REFMGR_KINDTYPE(lib) kind) {             \
		REFMGR_OBJTYPE(lib) *obj = (REFMGR_OBJTYPE(lib) *)                     \
			malloc(sizeof(REFMGR_OBJTYPE(lib)));                               \
		objects.push_back(obj);                                                \
		obj->kind = kind;                                                      \
		obj->refcount = 0;                                                     \
		obj->handle = (kind << REFMGR_KIND_SHIFT) | objects.size();            \
		return obj;                                                            \
	}

#define REFMGR_GEN_OBJGET(lib)	                                               \
  int BOOST_PP_CAT(lib, _object_get)(int handle,                               \
                                         REFMGR_OBJTYPE(lib) * *ret) {         \
    std::size_t i = handle & REFMGR_KEY_MASK;                                  \
    REFMGR_OBJTYPE(lib) * obj;                                                 \
    if ((i > objects.size()) or (i == 0))                                      \
      obj = nullptr;                                                           \
    else                                                                       \
      obj = objects[i - 1];                                                    \
    *ret = obj;                                                                \
    if (obj) {                                                                 \
      if (obj->kind != REFMGR_GET_KIND(handle)) {                              \
        *ret = nullptr;                                                        \
        return REFMGR_ERR_KIND;                                                \
      }                                                                        \
    } else {                                                                   \
      return REFMGR_ERR_OTHER;                                                 \
    }                                                                          \
    return REFMGR_SUCCESS;                                                     \
  }

#define REFMGR_GEN_INCREF(lib)                                                 \
  int BOOST_PP_CAT(lib, _object_incref)(int handle) {                          \
    REFMGR_OBJTYPE(lib) * obj;                                                 \
    BOOST_PP_CAT(lib, _object_get)(handle, &obj);                              \
    if (obj)                                                                   \
      obj->refcount++;                                                         \
    else                                                                       \
      return 1;                                                                \
    return 0;                                                                  \
  }

#define REFMGR_GEN_DECREF(lib)                                                 \
	int BOOST_PP_CAT(lib, _object_decref)(int handle) {                        \
		REFMGR_OBJTYPE(lib) *obj;                                              \
		BOOST_PP_CAT(lib, _object_get)(handle, &obj);                          \
		if (obj) {                                                             \
			if (obj->refcount > 0)                                             \
				obj->refcount--;                                               \
		} else return 1;                                                       \
		return 0;                                                              \
	}

#define REFMGR_GEN_FREE(lib)                                                   \
	int BOOST_PP_CAT(lib, _object_free)(int handle) {                          \
		REFMGR_OBJTYPE(lib) *obj;                                              \
		BOOST_PP_CAT(lib, _object_get)(handle, &obj);                          \
		if (obj) {                                                             \
			if (obj->refcount == 0) {                                          \
				free(obj);                                                     \
				objects[(handle & REFMGR_KEY_MASK) - 1] = nullptr;             \
			}                                                                  \
		} else return 0;                                                       \
		return 1;                                                              \
	}

#define REFMGR_GEN_FOOTER(libname) }

#define REFMGR_GEN(libname) \
	REFMGR_GEN_HEADER(libname) \
	REFMGR_GEN_OBJCREATE(libname) \
	REFMGR_GEN_OBJGET(libname) \
	REFMGR_GEN_INCREF(libname) \
	REFMGR_GEN_DECREF(libname) \
	REFMGR_GEN_FREE(libname) \
	REFMGR_GEN_FOOTER(libname)

#ifdef __cplusplus
}
#endif
