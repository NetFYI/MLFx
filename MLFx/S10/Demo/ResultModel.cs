using Microsoft.ML.Data;

namespace MLFx.S10.Demo
{
    class ResultModel
    {
        [ColumnName("Score")]
        public float Salary { get; set; }
    }
}
