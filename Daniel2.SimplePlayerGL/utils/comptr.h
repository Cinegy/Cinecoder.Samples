#ifndef __CC_COMPTR_H__
#define __CC_COMPTR_H__
#pragma once
// simple com smart pointer compatible (hope!) with stl containers
template<class I> class com_ptr
{
    typedef       com_ptr<I> &my_ref;
    typedef const com_ptr<I> &my_cref;

    class Tester { void operator delete(void*); };

protected:
	I * m_p;

public:
	com_ptr()               throw() : m_p(NULL)    { };
	com_ptr(I* pi)          throw() : m_p(pi)      { if (m_p != NULL) m_p->AddRef(); };
	com_ptr(my_cref ref)    throw() : m_p(ref.m_p) { if (m_p != NULL) m_p->AddRef(); };

	~com_ptr()              throw()	{ clear();     }

	// We do not allow implicit conversion to the original pointer type
	operator I*()     const throw()	{ return  m_p; }
	// Please use get() method to obtain the pointer - just to be sure what you are doing
	I* get()          const throw() { return  m_p; }
	void clear()                    { *this = 0;   }

	// For test-by-null purpose and to disallow the delete operator we introduce implicit conversion 
	// to the private type
	//operator Tester*()const throw() { if(!m_p) return 0; static Tester t; return &t; }

	// We also do not allow the address-of operator due to protection of ownership of pointer and also for
	// allowing to use the com_ptr with some of STL containers.
	//T** operator&()      { assert(m_pI == NULL); return &m_pI; }
	// For the factory methods which requires the address of the pointer we suggest the addr() method
	IUnknown** addr() const throw() { return (IUnknown**)&m_p; }

	I& operator*()    const throw()	{ return *m_p; }
	I* operator->()   const throw()	{ return  m_p; }

	I* operator=(I* pi)//     throw()
	{ 
	  // assignment from the itself - do nothing
	  if(pi == m_p)
		return m_p;

	  // At the first - increase the refcount of the new value
	  if(pi) 
	    pi->AddRef();

	  // Decrease refcount of the old value
	  if(m_p) 
	    m_p->Release();

	  // Assign the new value
	  return m_p = pi;
	}

	my_cref operator=(my_cref ref) throw()
	{ 
	  operator=(ref.m_p);
	  return *this;
	}
};

#endif //__CC_COMPTR_H__
