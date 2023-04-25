from gnn_student_teacher.data.structures import AbstractStudentResult
from gnn_student_teacher.data.structures import MockStudentResult


# == AbstractStudentResult ==

def test_mock_student_result_basically_works():
    result = MockStudentResult()
    assert isinstance(result, MockStudentResult)
    assert isinstance(result, AbstractStudentResult)

    # ~ Testing the normal dict interface
    value = 10
    result['value'] = value
    assert result['value'] == value

    result_dict = dict(result)
    assert isinstance(result_dict, dict)
    assert len(result_dict) == 1
    assert 'value' in result_dict

    # ~ Testing nested queries for the dict structure
    result['category/value'] = value
    assert result['category/value'] == value
    assert result['category']['value'] == value
