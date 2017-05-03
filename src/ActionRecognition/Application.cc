/*
 * Application.cc
 *
 *  Created on: Jun 29, 2015
 *      Author: richard
 */

#include "Application.hh"
#include <iostream>
#include "Example.hh"
#include "Stip.hh"

#include "Math/Random.hh"
#include "KMeans.hh"

using namespace ActionRecognition;

APPLICATION(ActionRecognition::Application)

const Core::ParameterEnum Application::paramAction_("action",
		"example, stip", // list of possible actions
		"example"); // the default

void Application::main() {

	switch (Core::Configuration::config(paramAction_)) {
	case example:
	{
		Example example;
		example.run();
	}
	break;
	case stip:
	{
		Stip stip;
		stip.run();
	}
	break;
	default:
		Core::Error::msg("No action given.") << Core::Error::abort;
	}
}
