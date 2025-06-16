#include <cstdlib> // For std::getopt
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <variant> // For std::variant

// Using a variant to handle both int and float values
using ConfigValue = std::variant<int, float>;

// Default values
std::map<std::string, ConfigValue> config;

// Function to read key-value pairs from the config file
void readConfigFile(const std::string &filename)
{
    std::ifstream configFile(filename);
    std::string line;
    if (configFile.is_open())
    {
        while (std::getline(configFile, line))
        {
            std::istringstream lineStream(line);
            std::string key, value;
            if (std::getline(lineStream, key, '='))
            {
                if (std::getline(lineStream, value))
                {
                    try
                    {
                        // Try to parse as an integer first
                        int intValue = std::stoi(value);
                        config[key] = intValue;
                    }
                    catch (const std::invalid_argument &)
                    {
                        try
                        {
                            // If parsing as an integer fails, try parsing as a float
                            float floatValue = std::stof(value);
                            config[key] = floatValue;
                        }
                        catch (const std::invalid_argument &)
                        {
                            std::cerr << "Warning: Invalid value for key " << key << ": " << value << std::endl;
                        }
                    }
                }
            }
        }
        configFile.close();
    }
    else
    {
        std::cerr << "Could not open config file: " << filename << std::endl;
    }
}

int main(int argc, char *argv[])
{
    // Default config file
    std::string configFile = "config.txt";

    int opt;
    // Parse command line arguments for the config file
    while ((opt = getopt(argc, argv, "c:")) != -1)
    {
        switch (opt)
        {
        case 'c':
            configFile = optarg; // Get the config file from the argument
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " [-c <config.txt>]" << std::endl;
            return 1;
        }
    }

    // Read the configuration file
    readConfigFile(configFile);

    // Print out config settings (for debugging)
    std::cout << "Config settings:" << std::endl;
    for (const auto &entry : config)
    {
        std::cout << entry.first << " = ";
        if (std::holds_alternative<int>(entry.second))
        {
            std::cout << std::get<int>(entry.second) << std::endl;
        }
        else if (std::holds_alternative<float>(entry.second))
        {
            std::cout << std::get<float>(entry.second) << std::endl;
        }
    }

    // Check the value of 'use_file' from config (defaults to 0 if not set)
    bool useFile = false;
    if (config.find("use_file") != config.end())
    {
        if (std::holds_alternative<int>(config["use_file"]) && std::get<int>(config["use_file"]) == 1)
        {
            useFile = true;
        }
        else if (std::holds_alternative<float>(config["use_file"]) && std::get<float>(config["use_file"]) == 1.0f)
        {
            useFile = true;
        }
    }

    // You can now access other config values (e.g., numeric values), for example:
    if (config.find("someKey") != config.end())
    {
        if (std::holds_alternative<int>(config["someKey"]))
        {
            std::cout << "someKey = " << std::get<int>(config["someKey"]) << std::endl;
        }
        else if (std::holds_alternative<float>(config["someKey"]))
        {
            std::cout << "someKey = " << std::get<float>(config["someKey"]) << std::endl;
        }
    }

    // Proceed with using `device`...
    // Clean up after usage

    return 0;
}
