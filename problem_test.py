"""Unit tests for problem.py."""
import unittest

from absl.testing import absltest
import problem as pr


class ProblemTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.defs = pr.Definition.from_txt_file('defs.txt', to_dict=True)

  def test_orthocenter_no_translate(self):
    txt = 'a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c'  # pylint: disable=line-too-long

    # read the txt into pr.Problem object, do not change the name of points:
    p = pr.Problem.from_txt(txt, translate=False)

    # This is fed into the LM, translating from constructive to constrained:
    setup_str = p.setup_str_from_problem(ProblemTest.defs)

    self.assertEqual(
        setup_str,
        '{S} a : ; b : ; c : ; h : T a b c h 00 T a c b h 01 ? T a h b c',
    )

  def test_orthocenter_translate(self):
    txt = 'a b c = triangle a b c; h = on_tline h b a c, on_tline h c a b ? perp a h b c'  # pylint: disable=line-too-long

    # Read the txt into pr.Problem object, change h -> d to match
    # training data distribution.
    p = pr.Problem.from_txt(txt, translate=True)

    # This is fed into the LM, translating from constructive to constrained:
    setup_str = p.setup_str_from_problem(ProblemTest.defs)

    self.assertEqual(
        setup_str,
        '{S} a : ; b : ; c : ; d : T a b c d 00 T a c b d 01 ? T a d b c',
    )


if __name__ == '__main__':
  absltest.main()
