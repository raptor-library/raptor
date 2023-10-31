#include <stdlib.h>
#include <vector>

#include <cedar/capi.h>
#include <cedar/interface/object.h>

static std::vector<cedar_object*> objects;

extern "C"
{

	cedar_object *cedar_object_create(cedar_kind kind)
	{
		cedar_object *obj = (cedar_object*) malloc(sizeof(cedar_object));
		objects.push_back(obj);

		obj->kind = kind;
		obj->refcount = 0;
		obj->handle = (kind << CEDAR_KIND_SHIFT) | objects.size();

		return obj;
	}


	int cedar_object_get(int handle, cedar_object **ret)
	{
		std::size_t i = handle & CEDAR_KEY_MASK;
		cedar_object *obj;
		if ((i > objects.size()) or (i == 0))
			obj = nullptr;
		else
			obj = objects[i-1];

		*ret = obj;
		if (obj) {
			if (obj->kind != CEDAR_GET_KIND(handle)) {
				*ret = nullptr;
				return CEDAR_ERR_KIND;
			}
		} else {
			return CEDAR_ERR_OTHER;
		}

		return CEDAR_SUCCESS;
	}


	int cedar_object_incref(int handle)
	{
		cedar_object *obj;
		cedar_object_get(handle, &obj);
		if (obj) {
			obj->refcount++;
		} else
			return 1;

		return 0;
	}


	int cedar_object_decref(int handle)
	{
		cedar_object *obj;
		cedar_object_get(handle, &obj);
		if (obj) {
			if (obj->refcount > 0)
				obj->refcount--;
		} else
			return 1;

		return 0;
	}


	int cedar_object_free(int handle)
	{
		cedar_object *obj;
		cedar_object_get(handle, &obj);
		if (obj) {
			if (obj->refcount == 0) {
				free(obj);
				objects[(handle & CEDAR_KEY_MASK) - 1] = nullptr;
			}
		} else {
			return 0;
		}

		return 1;
	}
}
