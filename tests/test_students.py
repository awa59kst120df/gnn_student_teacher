import tensorflow as tf
import tensorflow.keras as ks

from gnn_student_teacher.students import AbstractStudentModel
from gnn_student_teacher.students import StudentTemplate


# == AbstractStudentModel ==

class LocalStudent(AbstractStudentModel):

    def __init__(self, name, variant, **kwargs):
        super(LocalStudent, self).__init__(name=name, variant=variant, **kwargs)


def test_abstract_student_construction_works():
    student = LocalStudent('test', 'ref')
    assert isinstance(student, LocalStudent)
    assert isinstance(student, AbstractStudentModel)
    assert student.full_name == f'test_ref'


# == StudentTemplate ==

def test_student_template_construction_works():
    template = StudentTemplate(LocalStudent, 'test')
    assert isinstance(template, StudentTemplate)

    # From this template we should be able to instantiate actually distinct student objects. It is very
    # important that they are not the same object! That is the whole point of having the template mechanic
    student_ref = template.instantiate('ref')
    student_exp = template.instantiate('exp')
    assert isinstance(student_ref, AbstractStudentModel)
    assert isinstance(student_exp, AbstractStudentModel)
    assert student_ref.full_name != student_exp.full_name
    assert student_ref != student_exp




