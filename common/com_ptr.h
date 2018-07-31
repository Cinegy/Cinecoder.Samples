#ifndef __COM_PTR_H__
#define __COM_PTR_H__

#pragma once
// simple COM smart pointer
template<class I> class com_ptr
{
    typedef       com_ptr<I> &my_ref;
    typedef const com_ptr<I> &my_cref;

protected:
	I * m_p;

public:
	com_ptr()               throw() : m_p(NULL)    { };
	com_ptr(I* pi)          throw() : m_p(pi)      { if (m_p != NULL) m_p->AddRef(); };
	com_ptr(my_cref ref)    throw() : m_p(ref.m_p) { if (m_p != NULL) m_p->AddRef(); };

	~com_ptr()              throw()	{ clear();     }

	void clear()                    { *this = 0;   }

	I** operator&()      			{ return &m_p; }
	operator I*()     const throw()	{ return  m_p; }
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

#endif //__COM_PTR_H__
