using Microsoft.ML.Data;

namespace MLFx.S11.Demo
{
    class ResultModel
    {
        [ColumnName("Score")]
        public float Salary { get; set; }   
    }
}
