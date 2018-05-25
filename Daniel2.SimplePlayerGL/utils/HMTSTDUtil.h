#pragma once

#include <thread>
#include <mutex>
#include <list>
#include <chrono>

#define INFINITE_TIME 0xFFFFFFFF  // Infinite timeout

namespace cinegy
{
	namespace threading_std
	{
		class C_CritSec
		{
			C_CritSec(const C_CritSec &); // make copy constructor and assignment operator inaccessible
			C_CritSec &operator=(const C_CritSec &);

		public:
			C_CritSec() { }
			~C_CritSec() { }

			void Lock()
			{
				m_mutex.lock();
			}

			void Unlock()
			{
				m_mutex.unlock();
			}

		protected:
			std::mutex m_mutex;
		};

		class C_AutoLockMutex
		{
		private:
			std::unique_ptr<std::unique_lock<std::mutex>> m_lck;
		public:
			C_AutoLockMutex(std::mutex * mtx)
			{
				m_lck = std::make_unique<std::unique_lock<std::mutex>>(*mtx, std::defer_lock);

				if (m_lck)
					m_lck->lock();
			}
			~C_AutoLockMutex()
			{
				if (m_lck)
					m_lck->unlock();
			}
		};

		class C_AutoLock
		{
			C_AutoLock(const C_AutoLock &); // make copy constructor and assignment operator inaccessible
			C_AutoLock &operator=(const C_AutoLock &);

		public:
			C_AutoLock(C_CritSec * plock) { m_pLock = plock; m_pLock->Lock(); }
			~C_AutoLock() { m_pLock->Unlock(); }

		protected:
			C_CritSec * m_pLock;
		};

		class C_Event
		{
			C_Event(const C_Event &);	// make copy constructor and assignment operator inaccessible
			C_Event &operator=(const C_Event &);

			C_Event(C_Event &&ev);
			C_Event &operator=(C_Event &&ev);

		public:
			C_Event() { m_notified = false; }

			void Set() { m_notified = true; m_cond_var.notify_one(); }

			bool Wait(long long time = INFINITE_TIME)
			{
				std::unique_lock<std::mutex> lock(m_mutex);

				if (m_notified) return true;

				while (!m_notified) // loop to avoid spurious wakeups
				{
					if (time == INFINITE_TIME)
					{
						m_cond_var.wait(lock);
					}
					else
					{
						std::cv_status status = m_cond_var.wait_for(lock, std::chrono::milliseconds(time));

						if (status == std::cv_status::timeout)
							return false;
					}
				}

				return m_notified;
			}
			void Reset() { m_notified = false; }
			bool Check()
			{
				return m_notified;
			}

		protected:
			std::condition_variable m_cond_var;
			std::mutex m_mutex;
			bool m_notified;
		};

		template<class T> class C_SimpleThread
		{
			C_SimpleThread(const C_SimpleThread &);	// make copy constructor and assignment operator inaccessible
			C_SimpleThread &operator=(const C_SimpleThread &);

			C_SimpleThread(C_SimpleThread &&st);
			C_SimpleThread &operator=(C_SimpleThread &&st);

		public:
			C_SimpleThread() { m_thread = nullptr; m_evExit.Reset(); }
			~C_SimpleThread() { Close(); }

		protected:
			C_Event m_evExit;
			std::unique_ptr<std::thread> m_thread;

		//public:
		protected:
			long Create()
			{
				if (ThreadExists()) return 0;
				m_evExit.Reset();
				m_thread = std::make_unique<std::thread>(InitialThreadProc, (T *)this);
				if (ThreadExists()) return 0;
				//return HRESULT_FROM_WIN32(::GetLastError());
				return 0;
			}

			void Close()
			{
				if (ThreadExists())
				{
					m_evExit.Set();

					if (m_thread && m_thread->joinable())
					{
						//m_thread->detach();
						m_thread->join();
					}

					m_thread = nullptr;
				}
			}

			bool ThreadExists(void)
			{
				return (m_thread && !m_evExit.Check());
				//return (m_thread != nullptr);
			}

		protected:
			static long/*DWORD WINAPI*/ InitialThreadProc(void * p) { return ((T *)p)->ThreadProc(); }
			//DWORD ThreadProc();
		};

		template<class T> class C_BasicQueueT
		{
		public:
			C_BasicQueueT() : m_pEvSignal(NULL) { }
			C_BasicQueueT(C_Event * pEvSignal) : m_pEvSignal(pEvSignal) { }

			void SetSignal(C_Event * pEvSignal) { m_pEvSignal = pEvSignal; }

		public:
			void Free()
			{
				C_AutoLock lock(&m_lock);

				for (typename elementList_t::iterator it = m_q.begin(); it != m_q.end(); it++) (*it)->Release();
				m_q.clear();
			}

			size_t GetCount() { C_AutoLock lock(&m_lock); return m_q.size(); }
			bool Empty() { C_AutoLock lock(&m_lock); return m_q.empty(); }

			bool SoftGet(T ** ppBlock, bool remove = true)
			{
				C_AutoLock lock(&m_lock);

				*ppBlock = NULL;

				if (m_q.empty()) return false;

				*ppBlock = m_q.back();
				if (remove) m_q.pop_back();
				else (*ppBlock)->AddRef();

				return true;
			}

			void Queue(T * pBlock)
			{
				C_AutoLock lock(&m_lock);

				pBlock->AddRef();
				m_q.push_front(pBlock);
				if (m_pEvSignal != NULL) m_pEvSignal->Set();
			}

		protected:
			C_CritSec m_lock;
			typedef std::list<T *> elementList_t;
			elementList_t m_q;

			C_Event * m_pEvSignal;
		};

		template<class T> class C_QueueT : public C_BasicQueueT<T>
		{
		public:
			C_QueueT() : C_BasicQueueT<T>(&m_evMySignal) { };

		public:
			bool Get(T ** ppBlock, C_Event & hExitEvent, bool remove = true)
			{
				if (hExitEvent.Check())
					return false;

				for (;;)
				{
					if (this->SoftGet(ppBlock, remove)) return true;
					if (hExitEvent.Wait(1) && m_evMySignal.Wait(1)) break;
				}

				return false;
			}

		protected:
			C_Event m_evMySignal;
		};
	}
}