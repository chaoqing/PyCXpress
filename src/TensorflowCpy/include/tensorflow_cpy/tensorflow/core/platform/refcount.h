#ifndef TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
#define TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_

#include <atomic>
#include <memory>

namespace tensorflow {
namespace core {

class RefCounted {
 public:
  // Initial reference count is one.
  RefCounted();

  // Increments reference count by one.
  void Ref() const;

  // Decrements reference count by one.  If the count remains
  // positive, returns false.  When the count reaches zero, returns
  // true and deletes this, in which case the caller must not access
  // the object afterward.
  bool Unref() const;

  // Gets the current reference count.
  int_fast32_t RefCount() const;

  // Return whether the reference count is one.
  // If the reference count is used in the conventional way, a
  // reference count of 1 implies that the current thread owns the
  // reference and no other thread shares it.
  // This call performs the test for a reference count of one, and
  // performs the memory barrier needed for the owning thread
  // to act on the object, knowing that it has exclusive access to the
  // object.
  bool RefCountIsOne() const;

 private:
  mutable std::atomic_int_fast32_t ref_;

  RefCounted(const RefCounted&) = delete;
  void operator=(const RefCounted&) = delete;
};

// A deleter class to form a std::unique_ptr that unrefs objects.
struct RefCountDeleter {
  void operator()(const RefCounted* o) const { o->Unref(); }
};

// A unique_ptr that unrefs the owned object on destruction.
template <typename T>
using RefCountPtr = std::unique_ptr<T, RefCountDeleter>;

}  // namespace core
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_REFCOUNT_H_
