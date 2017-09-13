#pragma once

#include<sol.hpp>
#include<iostream>
#include<string>

namespace DC
{
	using namespace std;

	class Config
	{
		const string m_script_name;
		sol::state lua;
	public:
		Config(const string& script_name):m_script_name(script_name)
		{
			lua.script_file(script_name);
			std::cout << "script name = " << script_name << std::endl;
		}
		virtual ~Config()
		{}

		template<typename T>
			T get(const string& name)
			{
				sol::optional<T> value = lua[name];
                if( value == sol::nullopt )
				{
					std::cout << "Error: No such property: " << name << std::endl;
					exit(0);
				} else std::cout << name << " = " << *value << std::endl;
				return *value;
			}
	};
}
