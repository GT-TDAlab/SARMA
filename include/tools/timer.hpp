#pragma once

#include <chrono>
#include <string>
#include <mutex>
#include <iostream>

namespace sarma {

	class timer {
		static std::mutex iomutex;
		const std::string name;
		const std::chrono::high_resolution_clock::time_point start;
		std::mutex &mtx;
		std::ostream &os;
	public:
		timer(std::string s, std::mutex &_mtx = iomutex, std::ostream &_os = std::cerr) : name(s), start(std::chrono::high_resolution_clock::now()), mtx(_mtx), os(_os) {
			std::lock_guard lock(mtx);
			#ifdef DEBUG
				os << name << " has started" << std::endl;
			#endif
		}
		~timer() {
			std::lock_guard lock(mtx);
			#ifdef DEBUG
				os << name << " took " << time() << "s" << std::endl;
			#endif
		}
		double time() const {
			return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
		}
	};

	std::mutex timer::iomutex;
}