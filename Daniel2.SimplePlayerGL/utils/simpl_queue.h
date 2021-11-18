#pragma once

#include <mutex>
#include <list>
#include <chrono>
#include <condition_variable>

#define INFINITE_TIME 0xFFFFFFFF  // Infinite timeout

namespace cinegy
{
	namespace simpl
	{
		class simpl_event
		{
		public:
			simpl_event() { notified_ = false; }

			void Set() 
			{ 
				std::lock_guard<std::mutex> lock(mutex_);

				notified_ = true; 
				cond_var_.notify_all(); 
			}

			bool Wait(long long timeout = INFINITE_TIME)
			{
				std::unique_lock<std::mutex> lock(mutex_);

				while (!notified_) // loop to avoid spurious wakeups
				{
					if (timeout == INFINITE_TIME)
					{
						cond_var_.wait(lock);
					}
					else
					{
						std::cv_status status = cond_var_.wait_for(lock, std::chrono::milliseconds(timeout));

						if (status == std::cv_status::timeout)
							return false;
					}
				}

				notified_ = false;

				return true;
			}
			
			void Reset() 
			{ 
				std::lock_guard<std::mutex> lock(mutex_); 

				notified_ = false; 
			}
			
			bool Check()
			{
				std::lock_guard<std::mutex> lock(mutex_);

				return notified_;
			}

		private:
			std::condition_variable cond_var_;
			std::mutex mutex_;
			bool notified_;
		};

		template<class T>
		class simpl_queue
		{
		public:
			simpl_queue() : completed_(false) {}

		public:
			void Free()
			{
				std::lock_guard<std::mutex> lock(mutex_);

				queue_.clear();
			}

			size_t GetCount()
			{
				std::lock_guard<std::mutex> lock(mutex_);

				return queue_.size();
			}

			bool Empty()
			{
				std::lock_guard<std::mutex> lock(mutex_);

				return queue_.empty();
			}

			void Complete()
			{
				std::lock_guard<std::mutex> lock(mutex_);

				completed_ = true;

				event_.Set();
			}

			bool Get(T **ppValue, bool remove = true, const long long timeout = INFINITE_TIME)
			{
				while (true)
				{
					{
						std::lock_guard<std::mutex> lock(mutex_);

						if (!queue_.empty())
						{
							*ppValue = queue_.back();
							if (remove) queue_.pop_back();

							event_.Set();
							return true;
						}
					}

					if (completed_)
						break;

					if (!event_.Wait(timeout))
						break;
				}

				return false;
			}

			void Queue(T *pValue)
			{
				std::lock_guard<std::mutex> lock(mutex_);

				queue_.push_front(pValue);

				event_.Set();
			}

		private:
			std::list<T*> queue_;

			std::mutex mutex_;
			simpl_event event_;

			bool completed_;
		};
	}
}
