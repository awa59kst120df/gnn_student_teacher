

class AbstractStudentResult:
    """
    This object represents the output of a *single* student model training process. Instances of concrete
    subclasses are supposed to be returned by the corresponding subclasses of
    :class:`gnn_student_teacher.students.AbstractStudentTrainingManager`.

    **Dict Interface**

    This class implements the *dict* interface, which means that it can mostly be used in the same manner
    as a dictionary. This interface should be used to add the relevant data to the object during the
    training process.

    .. code-block:: python

        from gnn_student_teacher.data import MockStudentResult

        result = MockStudentResult()
        result['value'] = 3
        print(result['value']) # 3

    **Nested Dictionary Queries**

    Additionally, the dict interface is extended to support accessing nested dict structures by using
    file-system like string paths as keys. In the case of setting values, the necessary nesting structure
    is automatically created, if it does not already exist

    .. code-block:: python

        from gnn_student_teacher.data import MockStudentResult

        result = MockStudentResult()
        result['category/value'] = 3
        print( result['category/value'] )  # 3
        print( result['category']['value'] )  # 3

    """
    def __init__(self):
        self.data = {}

    def validate(self):
        pass

    def to_dict(self) -> dict:
        return self.data

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __dict__(self) -> dict:
        return self.data

    def __setitem__(self, key, value):
        if isinstance(key, str):
            keys = key.split('/')
            current_dict = self.data
            for current_key in keys:
                if current_key != keys[-1]:
                    if current_key not in current_dict:
                        current_dict[current_key] = {}

                    current_dict = current_dict[current_key]

            # Setting the actual value now into the nested dict
            current_dict[current_key] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            keys = key.split('/')
            current_dict = self.data
            for current_key in keys:
                if current_key not in current_dict:
                    raise KeyError(
                        f'Student result does not contain the nested structure "{key}" and thus the value '
                        f'cannot be retrieved. The error occurs at the nesting level "{current_key}"'
                    )

                current_dict = current_dict[current_key]

            return current_dict


class MockStudentResult(AbstractStudentResult):

    def __init__(self):
        super(MockStudentResult, self).__init__()