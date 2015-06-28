function(add_src_promisc dir)
  file(GLOB_RECURSE ${dir}_sources_ ${dir}/*.cpp)

  # remove tests
  set(EXCLUDE_DIR "/tests/")
  foreach (TMP_PATH ${${dir}_sources_})
    string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
    if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
      list (REMOVE_ITEM ${dir}_sources_ ${TMP_PATH})
    endif ()
  endforeach(TMP_PATH)

  set(${dir}_SOURCES ${${dir}_sources_} PARENT_SCOPE)
  file(GLOB_RECURSE ${dir}_includes_ ${dir}/*.hpp)
  set(${dir}_INCLUDES ${${dir}_includes_} PARENT_SCOPE)
endfunction(add_src_promisc)
