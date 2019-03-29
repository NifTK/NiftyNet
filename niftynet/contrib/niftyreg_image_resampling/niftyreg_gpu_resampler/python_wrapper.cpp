#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "resampleKernel.h"

#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <numpy/ndarrayobject.h>
#include <cassert>
#include <stdexcept>

#warning DEL ME
#include <iostream>
#include <iterator>

namespace {
  using namespace boost;
  using namespace python;

  class resampler_c {
  private:
    object m_image;
    object m_deformation;
    int m_interpolation;

  public:
    object run(void);

  public:
    resampler_c(const object &image, const object &deformation, const int interpolation);
  };

  nifti_image* extract_as_nifti(object &r_object) {
    PyArrayObject *p_ndarray = reinterpret_cast<PyArrayObject*>(r_object.ptr());
    nifti_image *p_nim = nifti_simple_init_nim();
    int nof_dims = PyArray_NDIM(p_ndarray);
    npy_intp *shape = PyArray_DIMS(p_ndarray);

    assert(PyArray_TYPE(p_ndarray) == NPY_FLOAT);

    if (nof_dims > 5) {
      throw std::runtime_error("Have an image with more than 5 dimensions.");
    }

    p_nim->dim[0] = nof_dims;
    std::transform(shape, shape + nof_dims, p_nim->dim + 1, [](const npy_intp d) -> int {
        return int(d);
      });
    std::fill(p_nim->pixdim + 1, p_nim->pixdim + 1 + nof_dims, 1.0);

#warning DEL ME
    std::cout << "Image dims: " << nof_dims << std::endl;
    std::cout << "Extracting: [";
    std::copy(shape, shape + nof_dims, std::ostream_iterator<npy_intp>(std::cout << "Extracting: [", ","));
    std::cout << "]\n";

    p_nim->datatype = NIFTI_TYPE_FLOAT32;
    p_nim->nbyper = sizeof(float);

    for (mat44 *p_xform: {&p_nim->sto_xyz, &p_nim->sto_ijk}) {
      for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
          p_xform->m[r][c] = float(r == c);
        }
    }

    nifti_update_dims_from_array(p_nim);

    p_nim->data = PyArray_DATA(p_ndarray);

    return p_nim;
  }

  object convert_to_ndarray_and_deallocate(nifti_image *p_image) {
    PyObject *p_ndarray;
    npy_intp dims[7];

    std::transform(p_image->dim + 1, p_image->dim + 1 + p_image->dim[0], dims, [](const int d) -> npy_intp {
        return npy_intp(d);
      });

    p_ndarray = PyArray_SimpleNew(p_image->dim[0], dims, NPY_FLOAT32);
    std::copy(reinterpret_cast<float*>(p_image->data), reinterpret_cast<float*>(p_image->data) + p_image->nvox, reinterpret_cast<float*>(PyArray_DATA((PyArray_GETCONTIGUOUS(reinterpret_cast<PyArrayObject*>(p_ndarray))))));

    nifti_image_free(p_image);

    return object(handle<>(p_ndarray));
  }

  void deallocate_nifti_wrapper(nifti_image *p_nim) {
    p_nim->data = nullptr;
    nifti_image_free(p_nim);
  }

  resampler_c::resampler_c(const object &image, const object &deformation, const int interpolation) : m_image(image), m_deformation(deformation), m_interpolation(interpolation) {}

  object resampler_c::run() {
    nifti_image *p_source = extract_as_nifti(m_image);
    nifti_image *p_deformation = extract_as_nifti(m_deformation);
    nifti_image *p_resampled = resample(*p_deformation, *p_source, m_interpolation, 0.f, false);

    deallocate_nifti_wrapper(p_source);
    deallocate_nifti_wrapper(p_deformation);

    return convert_to_ndarray_and_deallocate(p_resampled);
  }
}

BOOST_PYTHON_MODULE(niftyreg_gpu_resampler) {
  class_<resampler_c>("GPUImageResampling", "CUDA-enabled, fast image resampling for python.",
                      init<object, object, int>(args("image", "deformation", "interpolation"),
R"(:param image: the image to resample (to its own space)
:param deformation: deformation field matrix (numpy matrix, same spatial dimensions as image).
:param interpolation: NiftyReg interpolation code (0: nearest, 1: linear, 3: b-spline).)")
    )
    .def("run", &resampler_c::run,
R"(Performs the resampling operation.
:return: the resampled image)");
}
