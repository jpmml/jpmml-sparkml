package org.jpmml.sparkml.model;

import com.microsoft.ml.spark.LightGBMBooster;
import com.microsoft.ml.spark.LightGBMClassificationModel;
import org.dmg.pmml.Model;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ClassificationModelConverter;
import org.jpmml.sparkml.model.HasTreeOptions;
import org.jpmml.sparkml.BoosterUtil;

public class LightGBMClassificationModelConverter extends ClassificationModelConverter<LightGBMClassificationModel>
        implements HasTreeOptions {
    public LightGBMClassificationModelConverter(LightGBMClassificationModel model) {
        super(model);
    }

    @Override
    public Model encodeModel(Schema schema) {
        LightGBMClassificationModel model = getTransformer();
        LightGBMBooster booster = model.getModel();
        return BoosterUtil.encodeBinaryClassificationBooster(booster, schema);
    }
}
