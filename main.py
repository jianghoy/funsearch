from implementation import funsearch
from implementation.config import ProgramsDatabaseConfig, Config
import argparse
import dataclasses

# Generate a argparse where it takes all required params of Config and ProgramsDatabaseConfig, should set
# them to all optional

programs_database_config_fields = dataclasses.fields(ProgramsDatabaseConfig)
config_fields = dataclasses.fields(Config)

def parse_args():
    parser = argparse.ArgumentParser(description="Entry point of funsearch")

    # For each field in the dataclass, add a corresponding argument.
    # Use the field's type and default value.
    for field in programs_database_config_fields:
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None,  # Set default to None, will use dataclass default if not provided
            help=f"{field.name} (default: {field.default})",
            required=False,
        )
    for field in config_fields:
        if field.name == 'programs_database':
            continue
        parser.add_argument(
            f"--{field.name}",
            type=field.type,
            default=None,  # Set default to None, will use dataclass default if not provided
            help=f"{field.name} (default: {field.default})",
            required=False,
        )

    parser.add_argument(
        "--specification_file",
        "-s",
        type=str,
        help="""
        Path to the specification file. Specification contains the code to evolve and the code to run, 
        denoted using decorators @funsearch.evolve and @funsearch.run. Decorators in spec files are just
        for annotation purposes only. In fact they're disabled per code_manpulation.ProgramVisitor
        See specs_example_cap_set.py for an example.
        """,
        required=True,
    )
    parser.add_argument(
        "--test_inputs_file",
        "-t",
        type=str,
        help="""
        Path to the file containing the test inputs. Each line of the file is a test input, 
        and the test inputs are separated by new lines. See test_inputs_example_cap_set.txt for an example.
        """,
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use the __dict__ attribute of args to pass the arguments to the dataclass.
    # Filter out None values to let dataclass use its default values.
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    programs_database_args_dict = {k: v for k, v in args_dict.items() if k in programs_database_config_fields}
    # Initialize your data object
    programs_database = ProgramsDatabaseConfig(**programs_database_args_dict)

    config_args_dict = {k: v for k, v in args_dict.items() if k in config_fields}
    config = Config(programs_database=programs_database, **config_args_dict)
    with open(args.specification_file) as file:
        specification = file.read()
    with open(args.test_inputs_file) as file:
        test_inputs = file.read().splitlines()
    funsearch.main(specification, test_inputs, config)

# generate default
if __name__ == "__main__":
    main()
